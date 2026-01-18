"""
Agent Loop API 视图 (LangChain v1 重构版)

使用 LangChain v1 的 create_agent + middleware 模式，
替代原有的手动 AgentOrchestrator 循环。

核心变更：
- 使用 create_agent() 统一创建 Agent
- 使用 SummarizationMiddleware 自动处理上下文压缩
- 使用 HumanInTheLoopMiddleware 处理 HITL 审批
- 在流处理层检测工具调用，生成 step_start/step_complete 事件
- 支持 stream 参数控制流式/非流式输出
- SSE 事件格式与旧版保持兼容，前端无需修改
"""
import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from django.http import StreamingHttpResponse, JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.authentication import JWTAuthentication
from asgiref.sync import sync_to_async

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain.agents import create_agent
from wharttest_django.checkpointer import get_async_checkpointer

from .middleware_config import get_middleware_from_config
from .playwright_instructions import PLAYWRIGHT_SCRIPT_INSTRUCTION
from .stop_signal import should_stop, clear_stop_signal
from langgraph_integration.models import ChatSession, LLMConfig
from langgraph_integration.views import (
    create_llm_instance,
    create_sse_data,
    get_effective_system_prompt_async,
    check_project_permission,
)
from projects.models import Project
from prompts.models import UserPrompt
from mcp_tools.models import RemoteMCPConfig
from mcp_tools.persistent_client import mcp_session_manager
from requirements.context_limits import context_checker

logger = logging.getLogger(__name__)


# ============== 统一响应辅助函数 ==============

def api_success_response(message: str, data: Any = None, code: int = 200) -> JsonResponse:
    """构建统一格式的成功响应"""
    return JsonResponse({
        'status': 'success',
        'code': code,
        'message': message,
        'data': data,
        'errors': None
    }, json_dumps_params={'ensure_ascii': False})


def api_error_response(message: str, code: int = 400, errors: Any = None) -> JsonResponse:
    """构建统一格式的错误响应"""
    if errors is None:
        errors = {'detail': [message]}
    return JsonResponse({
        'status': 'error',
        'code': code,
        'message': message,
        'data': None,
        'errors': errors
    }, status=code, json_dumps_params={'ensure_ascii': False})


@method_decorator(csrf_exempt, name='dispatch')
class AgentLoopStreamAPIView(View):
    """
    Agent Loop 聊天 API (LangChain v1 重构版)

    核心特性：
    - 使用 create_agent() 统一创建 Agent
    - SummarizationMiddleware 自动上下文压缩
    - HumanInTheLoopMiddleware 处理 HITL 审批
    - 在流处理层检测工具调用生成步骤事件
    - 支持 stream 参数：
      - stream=true (默认)：返回 SSE 流式响应
      - stream=false：返回普通 JSON 响应
    """

    # 最大步骤数（用于前端显示）
    MAX_STEPS = 500

    async def authenticate_request(self, request):
        """JWT 认证"""
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise AuthenticationFailed('Authentication credentials were not provided.')

        token = auth_header.split(' ')[1]
        jwt_auth = JWTAuthentication()

        try:
            validated_token = await sync_to_async(jwt_auth.get_validated_token)(token)
            user = await sync_to_async(jwt_auth.get_user)(validated_token)
            return user
        except Exception as e:
            raise AuthenticationFailed(f'Invalid token: {str(e)}')

    async def _create_stream_generator(
        self,
        request,
        user_message: str,
        session_id: str,
        project_id: str,
        project: Project,
        knowledge_base_id: Optional[int] = None,
        use_knowledge_base: bool = True,
        prompt_id: Optional[int] = None,
        image_base64: Optional[str] = None,
        generate_playwright_script: bool = False,
        test_case_id: Optional[int] = None,
        use_pytest: bool = True,
    ):
        """
        创建 SSE 流式生成器（LangChain v1 重构版）

        使用 create_agent + astream 模式，替代旧的 AgentOrchestrator 循环。
        通过检测 updates 流中的工具调用来生成 step_start/step_complete 事件。
        """
        thread_id = f"{request.user.id}_{project_id}_{session_id}"

        # 1. 获取 LLM 配置
        try:
            active_config = await sync_to_async(LLMConfig.objects.get)(is_active=True)
            logger.info(f"AgentLoopStreamAPI: Using LLM config: {active_config.name}")
            context_limit = active_config.context_limit or 128000
            model_name = active_config.name or "gpt-4o"
        except LLMConfig.DoesNotExist:
            yield create_sse_data({'type': 'error', 'message': 'No active LLM configuration found'})
            return

        # 2. 验证多模态支持
        if image_base64 and not active_config.supports_vision:
            yield create_sse_data({
                'type': 'error',
                'message': f'模型 {active_config.name} 不支持图片输入'
            })
            return

        try:
            # 3. 初始化 LLM
            llm = await sync_to_async(create_llm_instance)(active_config, temperature=0.7)

            # 4. 加载 MCP 工具
            tools: List[Any] = []
            try:
                active_mcp_configs = await sync_to_async(list)(
                    RemoteMCPConfig.objects.filter(is_active=True)
                )
                if active_mcp_configs:
                    client_config = {}
                    for cfg in active_mcp_configs:
                        key = cfg.name or f"remote_{cfg.id}"
                        client_config[key] = {
                            "url": cfg.url,
                            "transport": (cfg.transport or "streamable_http").replace('-', '_'),
                        }
                        if cfg.headers:
                            client_config[key]["headers"] = cfg.headers

                    if client_config:
                        mcp_tools = await mcp_session_manager.get_tools_for_config(
                            client_config,
                            user_id=str(request.user.id),
                            project_id=str(project_id),
                            session_id=session_id
                        )
                        tools.extend(mcp_tools)
                        logger.info(f"AgentLoopStreamAPI: Loaded {len(mcp_tools)} MCP tools")
                        yield create_sse_data({
                            'type': 'info',
                            'message': f'已加载 {len(mcp_tools)} 个工具'
                        })
            except Exception as e:
                logger.error(f"AgentLoopStreamAPI: MCP tools loading failed: {e}", exc_info=True)
                yield create_sse_data({
                    'type': 'warning',
                    'message': f'MCP 工具加载失败: {str(e)}'
                })

            # 5. 添加知识库工具
            logger.info(f"AgentLoopStreamAPI: 检查知识库工具 - knowledge_base_id={knowledge_base_id}, use_knowledge_base={use_knowledge_base}")
            if knowledge_base_id and use_knowledge_base:
                try:
                    from knowledge.langgraph_integration import create_knowledge_tool
                    logger.info(f"AgentLoopStreamAPI: 正在创建知识库工具...")
                    kb_tool = await sync_to_async(create_knowledge_tool)(
                        knowledge_base_id=knowledge_base_id,
                        user=request.user
                    )
                    tools.append(kb_tool)
                    logger.info(f"AgentLoopStreamAPI: ✅ 知识库工具已添加: {kb_tool.name}")
                except Exception as e:
                    logger.warning(f"AgentLoopStreamAPI: ❌ Knowledge tool creation failed: {e}", exc_info=True)
            else:
                logger.info(f"AgentLoopStreamAPI: ⚠️ 跳过知识库工具 (knowledge_base_id={knowledge_base_id}, use_knowledge_base={use_knowledge_base})")

            # 6. 添加内置工具（Playwright 脚本管理等）
            from orchestrator_integration.builtin_tools import get_builtin_tools
            builtin_tools = get_builtin_tools(
                user_id=request.user.id,
                project_id=int(project_id),
                test_case_id=test_case_id,
                chat_session_id=session_id,
            )
            tools.extend(builtin_tools)
            logger.info(f"AgentLoopStreamAPI: Added {len(builtin_tools)} builtin tools")

            # 7. 获取或创建 ChatSession
            chat_session = await sync_to_async(
                lambda: ChatSession.objects.filter(
                    session_id=session_id,
                    user=request.user,
                    project_id=project_id
                ).first()
            )()
            
            if not chat_session:
                prompt_obj = None
                if prompt_id:
                    try:
                        prompt_obj = await sync_to_async(UserPrompt.objects.get)(
                            id=prompt_id, user=request.user, is_active=True
                        )
                    except UserPrompt.DoesNotExist:
                        pass

                chat_session = await sync_to_async(ChatSession.objects.create)(
                    user=request.user,
                    session_id=session_id,
                    project=project,
                    prompt=prompt_obj,
                    title=f"新对话 - {user_message[:30]}"
                )
                logger.info(f"AgentLoopStreamAPI: Created new ChatSession: {session_id}")

            # 8. 获取系统提示词
            effective_prompt, prompt_source = await get_effective_system_prompt_async(
                request.user, prompt_id, project
            )

            # 8.1 如果需要生成脚本，追加脚本生成指令
            if generate_playwright_script:
                effective_prompt = (effective_prompt or '') + PLAYWRIGHT_SCRIPT_INSTRUCTION
                logger.info(f"AgentLoopStreamAPI: 已追加脚本生成指令")

            # 9. 构建用户消息（支持多模态）
            if image_base64:
                human_message_content = [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            else:
                human_message_content = user_message
            user_msg = HumanMessage(content=human_message_content)

            # 10. 获取工具名列表用于 HITL
            tool_names = [t.name for t in tools] if tools else None

            # 11. 发送开始信号
            yield create_sse_data({
                'type': 'start',
                'session_id': session_id,
                'project_id': project_id,
                'mode': 'agent_loop',
                'created_at': chat_session.created_at.isoformat() if chat_session and chat_session.created_at else None
            })

            # 12. 创建 Agent（LangChain v1 统一路径）
            async with get_async_checkpointer() as checkpointer:
                # 获取中间件（需要同步到异步，因为内部有 ORM 查询）
                middleware = await sync_to_async(get_middleware_from_config)(
                    active_config, llm, user=request.user,
                    session_id=session_id, all_tool_names=tool_names
                )

                agent = create_agent(
                    llm,
                    tools,
                    system_prompt=effective_prompt,
                    checkpointer=checkpointer,
                    middleware=middleware,
                )
                logger.info(f"AgentLoopStreamAPI: Agent created with {len(tools)} tools")

                # 13. 配置调用参数
                invoke_config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 1000  # 支持约500次工具调用
                }
                input_messages = {"messages": [user_msg]}

                # 14. 步骤跟踪状态
                step_count = 0
                current_tool_calls = []
                interrupt_detected = False
                user_stopped = False

                # 15. 流式执行
                stream_modes = ["updates", "messages"]

                try:
                    async for stream_mode, chunk in agent.astream(
                        input_messages,
                        config=invoke_config,
                        stream_mode=stream_modes
                    ):
                        # 检查用户停止信号
                        if should_stop(session_id):
                            user_stopped = True
                            clear_stop_signal(session_id)
                            logger.info(f"AgentLoopStreamAPI: Stop signal received at step {step_count}")
                            yield create_sse_data({
                                'type': 'stopped',
                                'message': '已停止生成',
                                'step': step_count
                            })
                            break

                        if stream_mode == "updates":
                            # 检查中断事件 (HITL)
                            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                                interrupt_info = chunk["__interrupt__"]
                                logger.info(f"AgentLoopStreamAPI: HITL interrupt detected: {interrupt_info}")
                                logger.info(f"AgentLoopStreamAPI: interrupt_info type: {type(interrupt_info)}")

                                action_requests = []
                                interrupt_id = None
                                # 处理 tuple、list 或单个 Interrupt 对象
                                if isinstance(interrupt_info, (list, tuple)):
                                    interrupts_list = list(interrupt_info)
                                else:
                                    interrupts_list = [interrupt_info]

                                for intr in interrupts_list:
                                    logger.info(f"AgentLoopStreamAPI: Processing interrupt: type={type(intr)}, dir={dir(intr)}")
                                    logger.info(f"AgentLoopStreamAPI: interrupt repr: {repr(intr)}")

                                    if hasattr(intr, 'id'):
                                        interrupt_id = intr.id
                                        logger.info(f"AgentLoopStreamAPI: interrupt_id from attr: {interrupt_id}")
                                    elif isinstance(intr, dict) and 'id' in intr:
                                        interrupt_id = intr['id']
                                        logger.info(f"AgentLoopStreamAPI: interrupt_id from dict: {interrupt_id}")

                                    intr_value = getattr(intr, 'value', intr) if hasattr(intr, 'value') else intr
                                    logger.info(f"AgentLoopStreamAPI: intr_value type={type(intr_value)}, value={intr_value}")

                                    # 尝试多种方式获取 action_requests
                                    ars = []
                                    if isinstance(intr_value, dict):
                                        ars = intr_value.get('action_requests', [])
                                        logger.info(f"AgentLoopStreamAPI: action_requests from dict: {ars}")
                                    elif hasattr(intr_value, 'action_requests'):
                                        ars = intr_value.action_requests
                                        logger.info(f"AgentLoopStreamAPI: action_requests from attr: {ars}")

                                    # 如果还是空的，尝试从 intr 本身获取
                                    if not ars and hasattr(intr, 'action_requests'):
                                        ars = intr.action_requests
                                        logger.info(f"AgentLoopStreamAPI: action_requests from intr attr: {ars}")

                                    logger.info(f"AgentLoopStreamAPI: Found {len(ars)} action_requests: {ars}")

                                    for ar in ars:
                                        if isinstance(ar, dict):
                                            action_requests.append({
                                                'name': ar.get('name', ar.get('action_name', 'unknown')),
                                                'args': ar.get('arguments', ar.get('args', {})),
                                                'description': ar.get('description', ''),
                                            })
                                        else:
                                            action_requests.append({
                                                'name': getattr(ar, 'name', 'unknown'),
                                                'args': getattr(ar, 'arguments', getattr(ar, 'args', {})),
                                                'description': getattr(ar, 'description', ''),
                                            })

                                if action_requests:
                                    interrupt_detected = True
                                    yield create_sse_data({
                                        'type': 'interrupt',
                                        'interrupt_id': interrupt_id or str(id(interrupt_info)),
                                        'action_requests': action_requests,
                                        'session_id': session_id,
                                        'thread_id': thread_id,
                                    })
                                    logger.info(f"AgentLoopStreamAPI: Sent interrupt with {len(action_requests)} actions")

                            # 检测工具调用开始（用于生成 step_start 事件）
                            elif isinstance(chunk, dict):
                                for node_name, node_output in chunk.items():
                                    if node_name == "agent" and isinstance(node_output, dict):
                                        messages = node_output.get("messages", [])
                                        for msg in messages:
                                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                                # 新的工具调用 -> 新步骤开始
                                                step_count += 1
                                                current_tool_calls = msg.tool_calls
                                                tool_names_in_step = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in current_tool_calls]
                                                yield create_sse_data({
                                                    'type': 'step_start',
                                                    'step': step_count,
                                                    'max_steps': self.MAX_STEPS,
                                                    'tools': tool_names_in_step
                                                })
                                                logger.info(f"AgentLoopStreamAPI: Step {step_count} started with tools: {tool_names_in_step}")

                                    elif node_name == "tools" and isinstance(node_output, dict):
                                        # 工具执行完成
                                        tool_messages = node_output.get("messages", [])
                                        for tool_msg in tool_messages:
                                            if hasattr(tool_msg, 'content'):
                                                content = tool_msg.content
                                                summary = content[:200] if isinstance(content, str) else str(content)[:200]
                                                yield create_sse_data({
                                                    'type': 'tool_result',
                                                    'summary': summary,
                                                    'step': step_count
                                                })
                                        # 步骤完成
                                        if step_count > 0:
                                            yield create_sse_data({
                                                'type': 'step_complete',
                                                'step': step_count
                                            })

                        elif stream_mode == "messages":
                            # LLM Token 流式输出
                            # messages 模式返回元组 (token, metadata)
                            if isinstance(chunk, tuple) and len(chunk) >= 1:
                                token = chunk[0]
                                # 只发送 AI 消息，过滤掉 ToolMessage（工具结果已通过 tool_result 事件发送）
                                if hasattr(token, 'content') and token.content:
                                    # 检查是否是 ToolMessage（通过类名或 type 属性）
                                    token_type = type(token).__name__
                                    if 'ToolMessage' not in token_type:
                                        yield create_sse_data({'type': 'stream', 'data': token.content})
                            elif hasattr(chunk, 'content') and chunk.content:
                                # 兼容旧版本可能直接返回 message 的情况
                                # 同样过滤掉 ToolMessage
                                chunk_type = type(chunk).__name__
                                if 'ToolMessage' not in chunk_type:
                                    yield create_sse_data({'type': 'stream', 'data': chunk.content})

                except Exception as e:
                    logger.error(f"AgentLoopStreamAPI: Streaming error: {e}", exc_info=True)
                    yield create_sse_data({'type': 'error', 'message': f'Streaming error: {str(e)}'})

                # 16. 处理结束状态
                if user_stopped:
                    yield create_sse_data({
                        'type': 'complete',
                        'status': 'stopped',
                        'steps': step_count
                    })
                elif interrupt_detected:
                    logger.info("AgentLoopStreamAPI: Interrupt detected, returning early")
                else:
                    # 正常完成
                    # 计算 Token 使用量（使用 LangChain 的 count_tokens_approximately 保持一致性）
                    try:
                        current_state = await agent.aget_state(invoke_config)
                        all_messages = current_state.values.get("messages", []) if current_state.values else []
                        total_tokens = count_tokens_approximately(all_messages)

                        yield create_sse_data({
                            'type': 'context_update',
                            'context_token_count': total_tokens,
                            'context_limit': context_limit
                        })
                    except Exception as e:
                        logger.warning(f"AgentLoopStreamAPI: Failed to calculate token count: {e}")

                    complete_data = {
                        'type': 'complete',
                        'total_steps': step_count
                    }
                    if generate_playwright_script:
                        complete_data['script_generation'] = {
                            'enabled': True,
                            'message': '脚本管理工具已启用'
                        }
                    yield create_sse_data(complete_data)

                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"AgentLoopStreamAPI: Error: {e}", exc_info=True)
            yield create_sse_data({
                'type': 'error',
                'message': f'执行错误: {str(e)}'
            })

    async def post(self, request, *args, **kwargs):
        """
        处理聊天请求

        支持 stream 参数：
        - stream=true (默认)：返回 SSE 流式响应
        - stream=false：返回普通 JSON 响应
        """
        # 1. 认证
        try:
            user = await self.authenticate_request(request)
            request.user = user
        except AuthenticationFailed as e:
            return api_error_response(str(e), 401)

        # 2. 解析请求
        try:
            body_data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return api_error_response(f'Invalid JSON: {e}', 400)

        user_message = body_data.get('message')
        session_id = body_data.get('session_id')
        project_id = body_data.get('project_id')
        knowledge_base_id = body_data.get('knowledge_base_id')
        use_knowledge_base = body_data.get('use_knowledge_base', True)
        prompt_id = body_data.get('prompt_id')

        # 调试日志：知识库参数
        logger.info(f"AgentLoopStreamAPI: knowledge_base_id={knowledge_base_id}, use_knowledge_base={use_knowledge_base}")
        image_base64 = body_data.get('image')

        # stream 参数：控制流式/非流式输出（默认 true）
        stream_mode = body_data.get('stream', True)
        if isinstance(stream_mode, str):
            stream_mode = stream_mode.lower() in ('true', '1', 'yes')

        # Playwright 脚本生成参数
        generate_playwright_script = body_data.get('generate_playwright_script', False)
        test_case_id = body_data.get('test_case_id')  # 用于关联生成的脚本
        use_pytest = body_data.get('use_pytest', True)  # 生成 pytest 格式还是简单格式

        # 兜底：如果前端没传 test_case_id，尝试从消息中解析
        if not test_case_id and user_message:
            import re
            # 匹配 "执行ID为 11 的测试用例" 或 "测试用例 ID：11" 等模式
            match = re.search(r'(?:执行\s*ID\s*为|测试用例\s*(?:ID|id)[：:]\s*|case[_-]?id[：:=]\s*)(\d+)', user_message)
            if match:
                test_case_id = int(match.group(1))
                logger.info(f"AgentLoopStreamAPI: Parsed test_case_id from message: {test_case_id}")

        # 3. 参数验证
        if not project_id:
            return api_error_response('project_id is required', 400)

        if not user_message:
            return api_error_response('message is required', 400)

        # 4. 项目权限检查
        project = await sync_to_async(check_project_permission)(request.user, project_id)
        if not project:
            return api_error_response('Project access denied', 403)

        # 5. 生成 session_id
        if not session_id:
            session_id = uuid.uuid4().hex
            logger.info(f"AgentLoopStreamAPI: Generated new session_id: {session_id}")

        # 6. 根据 stream 参数决定响应方式
        if stream_mode:
            # 流式响应 (SSE)
            async def async_generator():
                async for chunk in self._create_stream_generator(
                    request, user_message, session_id, project_id, project,
                    knowledge_base_id, use_knowledge_base, prompt_id, image_base64,
                    generate_playwright_script, test_case_id, use_pytest
                ):
                    yield chunk

            response = StreamingHttpResponse(
                async_generator(),
                content_type='text/event-stream; charset=utf-8'
            )
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            return response
        else:
            # 非流式响应 (JSON)
            return await self._handle_non_stream_request(
                request, user_message, session_id, project_id, project,
                knowledge_base_id, use_knowledge_base, prompt_id, image_base64,
                generate_playwright_script, test_case_id, use_pytest
            )

    async def _handle_non_stream_request(
        self,
        request,
        user_message: str,
        session_id: str,
        project_id: str,
        project: Project,
        knowledge_base_id: Optional[int] = None,
        use_knowledge_base: bool = True,
        prompt_id: Optional[int] = None,
        image_base64: Optional[str] = None,
        generate_playwright_script: bool = False,
        test_case_id: Optional[int] = None,
        use_pytest: bool = True,
    ) -> JsonResponse:
        """
        处理非流式请求，收集所有流式事件后返回统一 JSON 响应
        """
        final_content = ""
        final_session_id = session_id
        tool_results = []
        total_steps = 0
        context_token_count = 0
        context_limit = 128000
        error_message = None
        interrupt_info = None
        script_generation = None

        try:
            async for chunk in self._create_stream_generator(
                request, user_message, session_id, project_id, project,
                knowledge_base_id, use_knowledge_base, prompt_id, image_base64,
                generate_playwright_script, test_case_id, use_pytest
            ):
                # 解析 SSE 数据
                if isinstance(chunk, str) and chunk.startswith('data: '):
                    data_str = chunk[6:].strip()
                    if data_str == '[DONE]':
                        continue
                    try:
                        event = json.loads(data_str)
                        event_type = event.get('type')

                        if event_type == 'start':
                            final_session_id = event.get('session_id', session_id)
                        elif event_type == 'stream':
                            # 累积流式内容
                            final_content += event.get('data', '')
                        elif event_type == 'tool_result':
                            tool_results.append({
                                'summary': event.get('summary', ''),
                                'step': event.get('step', 0)
                            })
                        elif event_type == 'step_complete':
                            total_steps = max(total_steps, event.get('step', 0))
                        elif event_type == 'context_update':
                            context_token_count = event.get('context_token_count', 0)
                            context_limit = event.get('context_limit', 128000)
                        elif event_type == 'error':
                            error_message = event.get('message', 'Unknown error')
                        elif event_type == 'interrupt':
                            interrupt_info = {
                                'interrupt_id': event.get('interrupt_id'),
                                'action_requests': event.get('action_requests', [])
                            }
                        elif event_type == 'complete':
                            if event.get('script_generation'):
                                script_generation = event.get('script_generation')
                    except json.JSONDecodeError:
                        continue

            # 构建响应
            if error_message:
                return api_error_response(error_message, 500)

            response_data = {
                'session_id': final_session_id,
                'content': final_content,
                'total_steps': total_steps,
                'tool_results': tool_results,
                'context_token_count': context_token_count,
                'context_limit': context_limit,
            }

            if interrupt_info:
                response_data['interrupt'] = interrupt_info

            if script_generation:
                response_data['script_generation'] = script_generation

            return api_success_response('Chat completed', response_data)

        except Exception as e:
            logger.error(f"AgentLoopStreamAPI: Non-stream request error: {e}", exc_info=True)
            return api_error_response(f'执行错误: {str(e)}', 500)


@method_decorator(csrf_exempt, name='dispatch')
class AgentLoopStopAPIView(View):
    """
    Agent Loop 停止 API

    用于中断正在执行的 Agent Loop 任务。
    """

    async def authenticate_request(self, request):
        """JWT 认证（复用 AgentLoopStreamAPIView 的逻辑）"""
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise AuthenticationFailed('Authentication credentials were not provided.')

        token = auth_header.split(' ')[1]
        jwt_auth = JWTAuthentication()

        try:
            validated_token = await sync_to_async(jwt_auth.get_validated_token)(token)
            user = await sync_to_async(jwt_auth.get_user)(validated_token)
            return user
        except Exception as e:
            raise AuthenticationFailed(f'Invalid token: {str(e)}')

    async def post(self, request, *args, **kwargs):
        """处理停止请求"""
        from .stop_signal import set_stop_signal

        # 1. 认证
        try:
            user = await self.authenticate_request(request)
            request.user = user
        except AuthenticationFailed as e:
            return api_error_response(str(e), 401)

        # 2. 解析请求
        try:
            body_data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return api_error_response(f'Invalid JSON: {e}', 400)

        session_id = body_data.get('session_id')
        if not session_id:
            return api_error_response('session_id is required', 400)

        # 3. 设置停止信号
        success = set_stop_signal(session_id)

        logger.info(f"AgentLoopStopAPI: Stop signal set for session {session_id} by user {user.id}")

        return api_success_response('已发送停止信号', {
            'session_id': session_id,
            'success': success
        })


@method_decorator(csrf_exempt, name='dispatch')
class AgentLoopResumeAPIView(View):
    """
    Agent Loop Resume API (SSE 流式版)

    用于恢复被 HITL 中断的 Agent Loop 任务。
    接收用户对工具调用的审批决策，然后通过 SSE 流式返回后续执行结果。

    这样前端可以像处理主流一样处理 resume 后的工具执行和 LLM 响应。
    """

    # 最大步骤数（与主流保持一致）
    MAX_STEPS = 500

    async def authenticate_request(self, request):
        """JWT 认证（复用 AgentLoopStreamAPIView 的逻辑）"""
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise AuthenticationFailed('Authentication credentials were not provided.')

        token = auth_header.split(' ')[1]
        jwt_auth = JWTAuthentication()

        try:
            validated_token = await sync_to_async(jwt_auth.get_validated_token)(token)
            user = await sync_to_async(jwt_auth.get_user)(validated_token)
            return user
        except Exception as e:
            raise AuthenticationFailed(f'Invalid token: {str(e)}')

    async def _create_resume_stream_generator(
        self,
        user,
        session_id: str,
        project_id: str,
        resume_data: dict,
        knowledge_base_id: Optional[str] = None,
        use_knowledge_base: bool = False,
    ):
        """
        创建 Resume SSE 流式生成器

        与主流的 _create_stream_generator 类似，但使用 Command(resume=...) 来恢复执行。
        """
        from langgraph.types import Command

        # 1. 解析 resume 数据
        interrupt_id = list(resume_data.keys())[0] if resume_data else None
        if not interrupt_id:
            yield create_sse_data({'type': 'error', 'message': 'Invalid resume data format'})
            return

        decision_info = resume_data[interrupt_id].get('decisions', [{}])[0]
        decision_type = decision_info.get('type', 'reject')

        # 获取工具调用数量（前端传递）
        action_count = resume_data[interrupt_id].get('action_count', 1)

        # 构建 resume 值 - HITL middleware 需要 decisions 格式
        # 为每个 pending 工具调用生成相同的决策
        resume_value = {"decisions": [{"type": decision_type} for _ in range(action_count)]}

        # 2. 发送 resume 开始信号
        yield create_sse_data({
            'type': 'resume_start',
            'session_id': session_id,
            'decision': decision_type
        })

        try:
            async with get_async_checkpointer() as checkpointer:
                # 3. 获取 LLM 配置
                active_config = await sync_to_async(
                    LLMConfig.objects.filter(is_active=True).first
                )()

                if not active_config:
                    yield create_sse_data({'type': 'error', 'message': '没有可用的 LLM 配置'})
                    return

                context_limit = active_config.context_limit or 128000
                model_name = active_config.name or "gpt-4o"
                llm = await sync_to_async(create_llm_instance)(active_config)

                # 4. 加载工具
                tools = []

                # 加载 MCP 工具
                try:
                    active_mcp_configs = await sync_to_async(list)(
                        RemoteMCPConfig.objects.filter(is_active=True)
                    )
                    if active_mcp_configs:
                        client_config = {}
                        for cfg in active_mcp_configs:
                            key = cfg.name or f"remote_{cfg.id}"
                            client_config[key] = {
                                "url": cfg.url,
                                "transport": (cfg.transport or "streamable_http").replace('-', '_'),
                            }
                            if cfg.headers:
                                client_config[key]["headers"] = cfg.headers

                        if client_config:
                            mcp_tools = await mcp_session_manager.get_tools_for_config(
                                client_config,
                                user_id=str(user.id),
                                project_id=str(project_id) if project_id else "0",
                                session_id=session_id
                            )
                            tools.extend(mcp_tools)
                            logger.info(f"AgentLoopResumeAPI: Loaded {len(mcp_tools)} MCP tools")
                except Exception as e:
                    logger.warning(f"AgentLoopResumeAPI: MCP tools loading failed: {e}")

                # 加载知识库工具
                if knowledge_base_id and use_knowledge_base:
                    try:
                        from knowledge.langgraph_integration import create_knowledge_tool
                        kb_tool = await sync_to_async(create_knowledge_tool)(
                            knowledge_base_id=knowledge_base_id,
                            user=user
                        )
                        tools.append(kb_tool)
                        logger.info(f"AgentLoopResumeAPI: ✅ 知识库工具已添加: {kb_tool.name}")
                    except Exception as e:
                        logger.warning(f"AgentLoopResumeAPI: ❌ Knowledge tool creation failed: {e}")

                # 加载内置工具
                try:
                    from orchestrator_integration.builtin_tools import get_builtin_tools
                    builtin_tools = get_builtin_tools(
                        user_id=user.id,
                        project_id=int(project_id) if project_id else 0,
                        test_case_id=None,
                        chat_session_id=session_id,
                    )
                    tools.extend(builtin_tools)
                    logger.info(f"AgentLoopResumeAPI: Added {len(builtin_tools)} builtin tools")
                except Exception as e:
                    logger.warning(f"AgentLoopResumeAPI: Builtin tools loading failed: {e}")

                # 5. 获取工具名列表和中间件配置
                tool_names = [t.name for t in tools] if tools else []
                middleware = await sync_to_async(get_middleware_from_config)(
                    active_config, llm,
                    user=user,
                    session_id=session_id,
                    all_tool_names=tool_names
                )

                # 6. 创建 agent
                agent = create_agent(
                    llm,
                    tools,
                    checkpointer=checkpointer,
                    middleware=middleware,
                )

                thread_id = f"{user.id}_{project_id}_{session_id}" if project_id else session_id
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 1000
                }

                # 7. 构建 Command 来 resume
                command = Command(resume=resume_value)

                # 8. 步骤跟踪状态
                step_count = 0
                interrupt_detected = False

                # 9. 流式执行
                try:
                    async for stream_mode, chunk in agent.astream(
                        command,
                        config=config,
                        stream_mode=["updates", "messages"]
                    ):
                        if stream_mode == "updates":
                            # 检查中断事件 (HITL) - resume 后可能又触发新的中断
                            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                                interrupt_info = chunk["__interrupt__"]
                                logger.info(f"AgentLoopResumeAPI: HITL interrupt detected after resume: {interrupt_info}")

                                action_requests = []
                                new_interrupt_id = None

                                if isinstance(interrupt_info, (list, tuple)):
                                    interrupts_list = list(interrupt_info)
                                else:
                                    interrupts_list = [interrupt_info]

                                for intr in interrupts_list:
                                    if hasattr(intr, 'id'):
                                        new_interrupt_id = intr.id
                                    elif isinstance(intr, dict) and 'id' in intr:
                                        new_interrupt_id = intr['id']

                                    intr_value = getattr(intr, 'value', intr) if hasattr(intr, 'value') else intr
                                    if isinstance(intr_value, dict):
                                        ars = intr_value.get('action_requests', [])
                                    elif hasattr(intr_value, 'action_requests'):
                                        ars = intr_value.action_requests
                                    else:
                                        ars = []

                                    for ar in ars:
                                        if isinstance(ar, dict):
                                            action_requests.append({
                                                'name': ar.get('name', ar.get('action_name', 'unknown')),
                                                'args': ar.get('arguments', ar.get('args', {})),
                                                'description': ar.get('description', ''),
                                            })
                                        else:
                                            action_requests.append({
                                                'name': getattr(ar, 'name', 'unknown'),
                                                'args': getattr(ar, 'arguments', getattr(ar, 'args', {})),
                                                'description': getattr(ar, 'description', ''),
                                            })

                                if action_requests:
                                    interrupt_detected = True
                                    yield create_sse_data({
                                        'type': 'interrupt',
                                        'interrupt_id': new_interrupt_id or str(id(interrupt_info)),
                                        'action_requests': action_requests,
                                        'session_id': session_id,
                                        'thread_id': thread_id,
                                    })

                            # 检测工具调用开始
                            elif isinstance(chunk, dict):
                                for node_name, node_output in chunk.items():
                                    if node_name == "agent" and isinstance(node_output, dict):
                                        messages = node_output.get("messages", [])
                                        for msg in messages:
                                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                                step_count += 1
                                                tool_names_in_step = [
                                                    tc.get('name', 'unknown') if isinstance(tc, dict)
                                                    else getattr(tc, 'name', 'unknown')
                                                    for tc in msg.tool_calls
                                                ]
                                                yield create_sse_data({
                                                    'type': 'step_start',
                                                    'step': step_count,
                                                    'max_steps': self.MAX_STEPS,
                                                    'tools': tool_names_in_step
                                                })

                                    elif node_name == "tools" and isinstance(node_output, dict):
                                        tool_messages = node_output.get("messages", [])
                                        for tool_msg in tool_messages:
                                            if hasattr(tool_msg, 'content'):
                                                content = tool_msg.content
                                                summary = content[:200] if isinstance(content, str) else str(content)[:200]
                                                yield create_sse_data({
                                                    'type': 'tool_result',
                                                    'summary': summary,
                                                    'step': step_count
                                                })
                                        if step_count > 0:
                                            yield create_sse_data({
                                                'type': 'step_complete',
                                                'step': step_count
                                            })

                        elif stream_mode == "messages":
                            # LLM Token 流式输出
                            # messages 模式返回元组 (token, metadata)
                            if isinstance(chunk, tuple) and len(chunk) >= 1:
                                token = chunk[0]
                                # 只发送 AI 消息，过滤掉 ToolMessage（工具结果已通过 tool_result 事件发送）
                                if hasattr(token, 'content') and token.content:
                                    # 检查是否是 ToolMessage（通过类名或 type 属性）
                                    token_type = type(token).__name__
                                    if 'ToolMessage' not in token_type:
                                        yield create_sse_data({'type': 'stream', 'data': token.content})
                            elif hasattr(chunk, 'content') and chunk.content:
                                # 兼容旧版本可能直接返回 message 的情况
                                # 同样过滤掉 ToolMessage
                                chunk_type = type(chunk).__name__
                                if 'ToolMessage' not in chunk_type:
                                    yield create_sse_data({'type': 'stream', 'data': chunk.content})

                except Exception as e:
                    logger.error(f"AgentLoopResumeAPI: Streaming error: {e}", exc_info=True)
                    yield create_sse_data({'type': 'error', 'message': f'Streaming error: {str(e)}'})

                # 10. 处理结束状态
                if interrupt_detected:
                    logger.info("AgentLoopResumeAPI: New interrupt detected after resume")
                else:
                    # 正常完成 - 计算 Token 使用量（使用 LangChain 的 count_tokens_approximately 保持一致性）
                    try:
                        current_state = await agent.aget_state(config)
                        all_messages = current_state.values.get("messages", []) if current_state.values else []
                        total_tokens = count_tokens_approximately(all_messages)

                        yield create_sse_data({
                            'type': 'context_update',
                            'context_token_count': total_tokens,
                            'context_limit': context_limit
                        })
                    except Exception as e:
                        logger.warning(f"AgentLoopResumeAPI: Failed to calculate token count: {e}")

                    yield create_sse_data({
                        'type': 'complete',
                        'total_steps': step_count,
                        'decision': decision_type
                    })

                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception(f"AgentLoopResumeAPI: Error in resume stream for session {session_id}")
            yield create_sse_data({'type': 'error', 'message': str(e)})

    async def post(self, request, *args, **kwargs):
        """处理 HITL resume 请求 - 返回 SSE 流式响应"""
        # 1. 认证
        try:
            user = await self.authenticate_request(request)
            request.user = user
        except AuthenticationFailed as e:
            return StreamingHttpResponse(
                iter([create_sse_data({'type': 'error', 'message': str(e), 'code': 401})]),
                content_type='text/event-stream; charset=utf-8',
                status=401
            )

        # 2. 解析请求
        try:
            body_data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return StreamingHttpResponse(
                iter([create_sse_data({'type': 'error', 'message': f'Invalid JSON: {e}', 'code': 400})]),
                content_type='text/event-stream; charset=utf-8',
                status=400
            )

        session_id = body_data.get('session_id')
        project_id = body_data.get('project_id')
        resume_data = body_data.get('resume', {})
        # 知识库参数（用于 resume 时重新加载知识库工具）
        knowledge_base_id = body_data.get('knowledge_base_id')
        use_knowledge_base = body_data.get('use_knowledge_base', False)

        if not session_id:
            return StreamingHttpResponse(
                iter([create_sse_data({'type': 'error', 'message': 'session_id is required', 'code': 400})]),
                content_type='text/event-stream; charset=utf-8',
                status=400
            )

        if not resume_data:
            return StreamingHttpResponse(
                iter([create_sse_data({'type': 'error', 'message': 'resume data is required', 'code': 400})]),
                content_type='text/event-stream; charset=utf-8',
                status=400
            )

        logger.info(f"AgentLoopResumeAPI: Resume request for session {session_id}, knowledge_base_id={knowledge_base_id}")

        # 3. 返回 SSE 流式响应
        async def async_generator():
            async for chunk in self._create_resume_stream_generator(
                user, session_id, project_id, resume_data,
                knowledge_base_id, use_knowledge_base
            ):
                yield chunk

        response = StreamingHttpResponse(
            async_generator(),
            content_type='text/event-stream; charset=utf-8'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
