"""
LangChain v1 中间件配置模块

提供统一的中间件配置，用于所有 Agent 创建。
包含：
- ModelRetryMiddleware: 模型调用重试（替代手动重试逻辑）
- ToolRetryMiddleware: 工具调用重试
- SummarizationMiddleware: 上下文摘要（替代 ConversationCompressor）
- HumanInTheLoopMiddleware: 人工审批（新增功能）
- 用户审批偏好：支持"记住审批选择"功能
"""

import logging
from typing import Callable, List, Optional, Dict, Any, Iterable

from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

from requirements.context_limits import context_checker, get_context_limit_from_llm

logger = logging.getLogger(__name__)


def _create_token_counter(model_name: str) -> Callable[[Iterable], int]:
    """
    创建基于 tiktoken 的精确 Token 计数器

    注意：SummarizationMiddleware 需要的 token_counter 签名是:
    Callable[[Iterable[MessageLikeRepresentation]], int]
    即接收消息列表，返回总 token 数
    """
    from langchain_core.messages.utils import count_tokens_approximately

    def token_counter(messages: Iterable) -> int:
        """计算消息列表的 token 总数"""
        try:
            # 使用 LangChain 官方的 count_tokens_approximately
            # 它会正确处理 content、tool_calls、role 等所有字段
            return count_tokens_approximately(messages)
        except Exception as e:
            logger.warning(f"Token 计数失败，使用粗略估算: {e}")
            # 回退方案
            total = 0
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    total += context_checker.count_tokens(content, model_name)
            return total

    return token_counter


# ============== 重试中间件配置 ==============

def get_model_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> ModelRetryMiddleware:
    """
    获取模型调用重试中间件

    替代 agent_loop.py 中的手动重试逻辑：
    - 自动指数退避重试
    - 支持连接错误、API 错误等

    Args:
        max_retries: 最大重试次数
        backoff_factor: 退避因子（每次重试等待时间倍增）
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        jitter: 是否添加随机抖动
    """
    return ModelRetryMiddleware(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        jitter=jitter,
    )


def get_tool_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    tools: Optional[List[str]] = None,
    on_failure: str = "continue",
) -> ToolRetryMiddleware:
    """
    获取工具调用重试中间件

    替代 graph.py 中 create_knowledge_tool 的手动重试逻辑

    Args:
        max_retries: 最大重试次数
        backoff_factor: 退避因子
        initial_delay: 初始延迟（秒）
        tools: 指定哪些工具需要重试（None 表示所有）
        on_failure: 失败后行为 ("continue" | "raise")
    """
    return ToolRetryMiddleware(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        tools=tools,
        on_failure=on_failure,
    )


# ============== 摘要中间件配置 ==============

def get_summarization_middleware(
    model=None,  # 可以是字符串或 BaseChatModel 实例
    trigger_tokens: int = 96000,  # 128k 的 75%
    keep_messages: int = 4,
    model_name: str = "gpt-4o",  # 用于精确 Token 计数
) -> Optional[SummarizationMiddleware]:
    """
    获取摘要中间件

    替代 ConversationCompressor 的手动上下文压缩逻辑

    Args:
        model: 用于生成摘要的模型（字符串或 BaseChatModel 实例）
               如果为 None，则返回 None（不使用摘要中间件）
        trigger_tokens: 触发摘要的 Token 阈值
        keep_messages: 保留最近的消息数量
        model_name: 模型名称，用于精确 Token 计数（基于 tiktoken）

    Returns:
        SummarizationMiddleware 实例，如果 model 为 None 则返回 None
    """
    if model is None:
        return None

    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", trigger_tokens),
        keep=("messages", keep_messages),
        token_counter=_create_token_counter(model_name),
    )


# ============== 人工审批中间件配置 ==============

# 默认需要人工审批的高风险工具
DEFAULT_HIGH_RISK_TOOLS = {
    # 自动化脚本执行
    "execute_script": {
        "allowed_decisions": ["approve", "reject"],
        "description": "自动化脚本执行需要审批",
    },
    # 测试用例批量操作
    "batch_delete_testcases": {
        "allowed_decisions": ["approve", "reject"],
        "description": "批量删除测试用例需要审批",
    },
    # 数据库写操作
    "execute_sql": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": "SQL 执行需要审批",
    },
    # ============== Playwright MCP 工具 ==============
    # 浏览器导航操作
    "playwright_navigate": {
        "allowed_decisions": ["approve", "reject"],
        "description": "浏览器导航需要审批",
    },
    # 页面点击操作
    "playwright_click": {
        "allowed_decisions": ["approve", "reject"],
        "description": "页面点击操作需要审批",
    },
    # 表单填写
    "playwright_fill": {
        "allowed_decisions": ["approve", "reject"],
        "description": "表单填写需要审批",
    },
    # 脚本执行
    "playwright_evaluate": {
        "allowed_decisions": ["approve", "reject"],
        "description": "JavaScript 脚本执行需要审批",
    },
}


def get_mcp_hitl_tools() -> Dict[str, Any]:
    """
    从 RemoteMCPConfig 动态加载需要 HITL 审批的工具

    Returns:
        Dict[tool_name, config]: 需要审批的工具配置
    """
    from mcp_tools.models import RemoteMCPConfig

    hitl_tools = {}

    mcp_configs = RemoteMCPConfig.objects.filter(is_active=True, require_hitl=True)

    for config in mcp_configs:
        # 如果 hitl_tools 为空，表示该 MCP 的所有工具都需要审批
        # 这种情况在实际触发时由 HumanInTheLoopMiddleware 的通配符逻辑处理
        # 这里我们用 mcp_name 作为前缀标识
        if config.hitl_tools:
            # 有指定工具列表
            for tool_name in config.hitl_tools:
                hitl_tools[tool_name] = {
                    "allowed_decisions": ["approve", "reject"],
                    "description": f"[{config.name}] {tool_name} 需要审批",
                }
        else:
            # 空列表表示该 MCP 所有工具都需要审批
            # 使用特殊标记，后续在中间件中处理
            hitl_tools[f"__mcp_all__{config.name}"] = {
                "allowed_decisions": ["approve", "reject"],
                "description": f"[{config.name}] 所有工具需要审批",
                "_mcp_name": config.name,
                "_all_tools": True,
            }

    logger.debug("从 MCP 配置加载 HITL 工具: %s", list(hitl_tools.keys()))
    return hitl_tools


def get_user_tool_approvals(user, session_id: Optional[str] = None) -> Dict[str, str]:
    """
    获取用户的工具审批偏好

    Args:
        user: Django User 对象
        session_id: 可选的会话ID，用于获取会话级别的偏好

    Returns:
        Dict[tool_name, policy]: 工具名到审批策略的映射
        例如: {"execute_script": "always_allow", "run_playwright": "ask_every_time"}
    """
    from langgraph_integration.models import UserToolApproval

    approvals = {}

    # 先获取永久偏好
    permanent_approvals = UserToolApproval.objects.filter(
        user=user,
        scope='permanent'
    ).values('tool_name', 'policy')

    for approval in permanent_approvals:
        approvals[approval['tool_name']] = approval['policy']

    # 如果有会话ID，会话级偏好覆盖永久偏好
    if session_id:
        session_approvals = UserToolApproval.objects.filter(
            user=user,
            scope='session',
            session_id=session_id
        ).values('tool_name', 'policy')

        for approval in session_approvals:
            approvals[approval['tool_name']] = approval['policy']

    return approvals


def build_dynamic_interrupt_on(
    base_tools: Dict[str, Any],
    user=None,
    session_id: Optional[str] = None,
    all_tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    根据用户偏好动态构建 interrupt_on 配置

    Args:
        base_tools: 基础的高风险工具配置
        user: Django User 对象（可选）
        session_id: 会话ID（可选）
        all_tool_names: 当前 Agent 的所有工具名列表（可选）
                        如果提供，则默认所有工具都需要审批

    Returns:
        动态生成的 interrupt_on 配置
    """
    # 获取用户偏好
    user_approvals = get_user_tool_approvals(user, session_id) if user else {}

    # 如果提供了工具名列表，默认所有工具都需要审批
    if all_tool_names:
        dynamic_config = {}
        for tool_name in all_tool_names:
            user_policy = user_approvals.get(tool_name)

            if user_policy == 'always_allow':
                # 用户选择"始终允许"，跳过审批
                logger.debug("工具 %s 已被用户设为始终允许，跳过审批", tool_name)
                continue  # 不加入 interrupt_on，即不审批
            elif user_policy == 'always_reject':
                # 用户选择"始终拒绝"，保持审批配置
                dynamic_config[tool_name] = {
                    "allowed_decisions": ["approve", "reject"],
                    "description": f"{tool_name} 需要审批",
                }
                logger.debug("工具 %s 已被用户设为始终拒绝", tool_name)
            else:
                # 默认需要审批
                # 优先使用 base_tools 中的配置（如有描述）
                if tool_name in base_tools:
                    dynamic_config[tool_name] = base_tools[tool_name]
                else:
                    dynamic_config[tool_name] = {
                        "allowed_decisions": ["approve", "reject"],
                        "description": f"{tool_name} 需要审批",
                    }

        return dynamic_config

    # 传统模式：只审批 base_tools 中的工具
    if user is None:
        return base_tools

    dynamic_config = {}

    for tool_name, tool_config in base_tools.items():
        user_policy = user_approvals.get(tool_name)

        if user_policy == 'always_allow':
            # 用户选择"始终允许"，跳过审批
            dynamic_config[tool_name] = False
            logger.debug("工具 %s 已被用户设为始终允许，跳过审批", tool_name)
        elif user_policy == 'always_reject':
            # 用户选择"始终拒绝"，仍需审批但可以在前端自动拒绝
            # 这里仍保持审批配置，让前端处理自动拒绝逻辑
            dynamic_config[tool_name] = tool_config
            logger.debug("工具 %s 已被用户设为始终拒绝", tool_name)
        else:
            # 默认或"每次询问"
            dynamic_config[tool_name] = tool_config

    return dynamic_config


def get_human_in_the_loop_middleware(
    interrupt_on: Optional[Dict[str, Any]] = None,
    description_prefix: str = "工具执行待审批",
    user=None,
    session_id: Optional[str] = None,
    include_mcp_tools: bool = True,
    all_tool_names: Optional[List[str]] = None,
) -> HumanInTheLoopMiddleware:
    """
    获取人工审批中间件

    用于高风险操作前暂停执行，等待用户确认。
    支持根据用户偏好动态跳过已"始终允许"的工具。

    Args:
        interrupt_on: 需要中断的工具配置，格式：
            {
                "tool_name": True,  # 所有决策类型
                "tool_name": {"allowed_decisions": ["approve", "reject"]},
                "tool_name": False,  # 不需要审批
            }
        description_prefix: 中断消息前缀
        user: Django User 对象，用于读取用户审批偏好
        session_id: 会话ID，用于读取会话级别的偏好
        include_mcp_tools: 是否包含 MCP 配置中标记需要审批的工具
        all_tool_names: 当前 Agent 的所有工具名列表（可选）
                        如果提供，则默认所有工具都需要审批，用户可通过偏好跳过
    """
    base_config = dict(interrupt_on or DEFAULT_HIGH_RISK_TOOLS)

    # 动态加载 MCP 配置的 HITL 工具
    if include_mcp_tools:
        mcp_tools = get_mcp_hitl_tools()
        base_config.update(mcp_tools)

    # 根据用户偏好动态调整配置
    config = build_dynamic_interrupt_on(base_config, user, session_id, all_tool_names)

    return HumanInTheLoopMiddleware(
        interrupt_on=config,
        description_prefix=description_prefix,
    )


# ============== 组合中间件 ==============

def get_standard_middleware(
    enable_model_retry: bool = True,
    enable_tool_retry: bool = True,
    enable_summarization: bool = True,
    enable_hitl: bool = False,  # 人工审批默认关闭，按需开启
    hitl_tools: Optional[Dict[str, Any]] = None,
    hitl_user=None,  # 用于动态读取用户审批偏好
    hitl_session_id: Optional[str] = None,  # 用于会话级别偏好
    hitl_all_tool_names: Optional[List[str]] = None,  # 所有工具名，用于默认审批所有工具
    summarization_model=None,  # 需显式传入 LLM 实例或模型名
    summarization_trigger_tokens: int = 96000,
    summarization_keep_messages: int = 4,
    model_name: str = "gpt-4o",  # 用于精确 Token 计数
) -> List:
    """
    获取标准中间件组合

    提供开箱即用的中间件配置，适用于大多数 Agent 场景

    Args:
        enable_model_retry: 是否启用模型重试
        enable_tool_retry: 是否启用工具重试
        enable_summarization: 是否启用摘要
        enable_hitl: 是否启用人工审批
        hitl_tools: 人工审批工具配置
        hitl_user: Django User 对象，用于读取用户审批偏好
        hitl_session_id: 会话ID，用于读取会话级别的偏好
        hitl_all_tool_names: 当前 Agent 的所有工具名，用于默认审批所有工具
        summarization_model: 摘要模型
        summarization_trigger_tokens: 摘要触发阈值
        summarization_keep_messages: 保留消息数
        model_name: 模型名称，用于精确 Token 计数

    Returns:
        中间件列表
    """
    middleware = []

    if enable_model_retry:
        middleware.append(get_model_retry_middleware())
        logger.debug("已添加 ModelRetryMiddleware")

    if enable_tool_retry:
        middleware.append(get_tool_retry_middleware())
        logger.debug("已添加 ToolRetryMiddleware")

    if enable_summarization and summarization_model is not None:
        summarization_mw = get_summarization_middleware(
            model=summarization_model,
            trigger_tokens=summarization_trigger_tokens,
            keep_messages=summarization_keep_messages,
            model_name=model_name,
        )
        if summarization_mw is not None:
            middleware.append(summarization_mw)
            logger.info("✅ 已添加 SummarizationMiddleware (trigger_tokens=%d, keep_messages=%d, model=%s)",
                        summarization_trigger_tokens, summarization_keep_messages, model_name)
        else:
            logger.warning("⚠️ SummarizationMiddleware 创建失败，返回 None")
    else:
        logger.info("⏭️ 跳过 SummarizationMiddleware: enable_summarization=%s, summarization_model=%s",
                    enable_summarization, summarization_model is not None)

    if enable_hitl:
        middleware.append(get_human_in_the_loop_middleware(
            interrupt_on=hitl_tools,
            user=hitl_user,
            session_id=hitl_session_id,
            all_tool_names=hitl_all_tool_names,
        ))
        logger.debug("已添加 HumanInTheLoopMiddleware (all_tools=%s)", bool(hitl_all_tool_names))

    return middleware


def get_brain_middleware() -> List:
    """
    获取 Brain Agent 专用中间件

    Brain Agent 负责路由决策，不需要摘要和 HITL
    """
    return [
        get_model_retry_middleware(max_retries=2),
    ]


def get_chat_middleware(summarization_model=None) -> List:
    """
    获取 Chat Agent 专用中间件

    Chat Agent 需要上下文摘要支持长对话

    Args:
        summarization_model: 用于摘要的模型（LLM 实例或模型名）
    """
    return get_standard_middleware(
        enable_hitl=False,
        summarization_model=summarization_model,
        summarization_trigger_tokens=96000,
        summarization_keep_messages=4,
    )


def get_requirement_middleware(summarization_model=None) -> List:
    """
    获取 Requirement Agent 专用中间件

    Args:
        summarization_model: 用于摘要的模型（LLM 实例或模型名）
    """
    return get_standard_middleware(
        enable_hitl=False,
        summarization_model=summarization_model,
        summarization_trigger_tokens=96000,
        summarization_keep_messages=4,
    )


def get_testcase_middleware(summarization_model=None) -> List:
    """
    获取 TestCase Agent 专用中间件

    Args:
        summarization_model: 用于摘要的模型（LLM 实例或模型名）
    """
    return get_standard_middleware(
        enable_hitl=False,
        summarization_model=summarization_model,
        summarization_trigger_tokens=96000,
        summarization_keep_messages=4,
    )


def get_automation_middleware(summarization_model=None) -> List:
    """
    获取自动化执行 Agent 专用中间件

    包含人工审批，用于高风险的脚本执行操作

    Args:
        summarization_model: 用于摘要的模型（LLM 实例或模型名）
    """
    return get_standard_middleware(
        enable_hitl=True,
        summarization_model=summarization_model,
        hitl_tools={
            "execute_script": {
                "allowed_decisions": ["approve", "reject"],
                "description": "自动化脚本执行需要审批",
            },
            "run_playwright": {
                "allowed_decisions": ["approve", "reject"],
                "description": "浏览器自动化操作需要审批",
            },
        },
    )


# ============== 从 LLMConfig 构建中间件 ==============

def get_middleware_from_config(
    llm_config,
    llm=None,
    agent_type: str = "standard",
    user=None,
    session_id: Optional[str] = None,
    all_tool_names: Optional[List[str]] = None,
) -> List:
    """
    从 LLMConfig 模型配置构建中间件列表

    统一的中间件构建入口，根据 LLMConfig 中的配置字段决定启用哪些中间件。

    Args:
        llm_config: LLMConfig 模型实例，包含以下关键字段：
            - enable_summarization: 是否启用上下文摘要
            - enable_hitl: 是否启用人工审批
            - context_limit: 上下文Token限制（用于计算摘要触发阈值）
            - name: 模型名称（用于精确 Token 计数）
        llm: 用于摘要的 LLM 实例（如果 enable_summarization=True 则必须提供）
        agent_type: Agent 类型，决定额外的中间件配置
            - "brain": Brain Agent，只需重试中间件
            - "standard": 标准 Agent
            - "automation": 自动化 Agent，强制启用 HITL
        user: Django User 对象，用于读取用户的审批偏好（"记住此选择"功能）
        session_id: 会话ID，用于读取会话级别的偏好
        all_tool_names: 当前 Agent 的所有工具名列表（可选）
                        如果提供，则默认所有工具都需要审批，用户可通过偏好跳过

    Returns:
        中间件列表

    Example:
        from langgraph_integration.models import LLMConfig
        from orchestrator_integration.middleware_config import get_middleware_from_config

        config = LLMConfig.objects.get(is_active=True)
        tool_names = [t.name for t in tools]  # 获取所有工具名
        middleware = get_middleware_from_config(config, llm=my_llm, user=request.user, all_tool_names=tool_names)
        agent = create_agent(llm, tools, middleware=middleware)
    """
    # Brain Agent 只需要轻量级重试
    if agent_type == "brain":
        return [get_model_retry_middleware(max_retries=2)]

    # 从 LLMConfig 读取配置
    enable_summarization = getattr(llm_config, 'enable_summarization', True)
    enable_hitl = getattr(llm_config, 'enable_hitl', False)
    model_name = getattr(llm_config, 'name', 'gpt-4o')

    # 上下文限制：优先使用 LLMConfig 中用户配置的值
    config_context_limit = getattr(llm_config, 'context_limit', None)
    if config_context_limit and config_context_limit > 0:
        context_limit = config_context_limit
    elif llm is not None:
        # 后备：从 LLM 的 Model Profile 获取
        context_limit = get_context_limit_from_llm(llm, fallback_model_name=model_name)
    else:
        context_limit = 128000

    # 自动化 Agent 强制启用 HITL
    if agent_type == "automation":
        enable_hitl = True

    # 计算摘要触发阈值（上下文限制的 75%）
    trigger_tokens = int(context_limit * 0.75)

    # 决定摘要模型
    summarization_model = llm if enable_summarization else None

    # HITL 工具配置
    hitl_tools = None
    if agent_type == "automation":
        hitl_tools = {
            "execute_script": {
                "allowed_decisions": ["approve", "reject"],
                "description": "自动化脚本执行需要审批",
            },
            "run_playwright": {
                "allowed_decisions": ["approve", "reject"],
                "description": "浏览器自动化操作需要审批",
            },
        }

    logger.info(
        "从 LLMConfig 构建中间件: agent_type=%s, summarization=%s, hitl=%s, context_limit=%d, trigger_tokens=%d, model=%s, user=%s, all_tools=%d",
        agent_type, enable_summarization, enable_hitl, context_limit, trigger_tokens, model_name,
        user.username if user else None, len(all_tool_names) if all_tool_names else 0
    )

    return get_standard_middleware(
        enable_model_retry=True,
        enable_tool_retry=True,
        enable_summarization=enable_summarization,
        enable_hitl=enable_hitl,
        hitl_tools=hitl_tools,
        hitl_user=user,
        hitl_session_id=session_id,
        hitl_all_tool_names=all_tool_names,
        summarization_model=summarization_model,
        summarization_trigger_tokens=trigger_tokens,
        summarization_keep_messages=4,
        model_name=model_name,
    )


# ============== 异步版本 ==============

async def get_mcp_hitl_tools_async() -> Dict[str, Any]:
    """获取 MCP 配置中标记为需要审批的工具（异步版本）"""
    from asgiref.sync import sync_to_async
    from mcp_tools.models import RemoteMCPConfig

    hitl_tools = {}

    @sync_to_async
    def get_mcp_configs():
        return list(RemoteMCPConfig.objects.filter(is_active=True, require_hitl=True))

    mcp_configs = await get_mcp_configs()

    for config in mcp_configs:
        if config.hitl_tools:
            for tool_name in config.hitl_tools:
                hitl_tools[tool_name] = {
                    "allowed_decisions": ["approve", "reject"],
                    "description": f"[{config.name}] {tool_name} 需要审批",
                }
        else:
            hitl_tools[f"__mcp_all__{config.name}"] = {
                "allowed_decisions": ["approve", "reject"],
                "description": f"[{config.name}] 所有工具需要审批",
                "_mcp_name": config.name,
                "_all_tools": True,
            }

    logger.debug("从 MCP 配置加载 HITL 工具: %s", list(hitl_tools.keys()))
    return hitl_tools


async def get_user_tool_approvals_async(user, session_id: Optional[str] = None) -> Dict[str, str]:
    """获取用户的工具审批偏好（异步版本）"""
    from asgiref.sync import sync_to_async
    from langgraph_integration.models import UserToolApproval

    approvals = {}

    @sync_to_async
    def get_permanent_approvals():
        return list(UserToolApproval.objects.filter(
            user=user,
            scope='permanent'
        ).values('tool_name', 'policy'))

    @sync_to_async
    def get_session_approvals():
        return list(UserToolApproval.objects.filter(
            user=user,
            scope='session',
            session_id=session_id
        ).values('tool_name', 'policy'))

    permanent_approvals = await get_permanent_approvals()
    for approval in permanent_approvals:
        approvals[approval['tool_name']] = approval['policy']

    if session_id:
        session_approvals = await get_session_approvals()
        for approval in session_approvals:
            approvals[approval['tool_name']] = approval['policy']

    return approvals


async def build_dynamic_interrupt_on_async(
    base_tools: Dict[str, Any],
    user=None,
    session_id: Optional[str] = None,
    all_tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """根据用户偏好动态构建 interrupt_on 配置（异步版本）"""
    config = dict(base_tools)

    if all_tool_names:
        for tool_name in all_tool_names:
            if tool_name not in config:
                config[tool_name] = {
                    "allowed_decisions": ["approve", "reject"],
                    "description": f"工具 {tool_name} 需要审批",
                }

    if not user:
        return config

    user_approvals = await get_user_tool_approvals_async(user, session_id)

    for tool_name, policy in user_approvals.items():
        if policy == "always_allow":
            config[tool_name] = False
            logger.debug("用户偏好: %s -> 始终允许，跳过审批", tool_name)
        elif policy == "always_reject":
            config[tool_name] = False
            logger.debug("用户偏好: %s -> 始终拒绝，跳过审批（调用将被阻止）", tool_name)

    return config


async def get_human_in_the_loop_middleware_async(
    interrupt_on: Optional[Dict[str, Any]] = None,
    description_prefix: str = "需要审批:",
    user=None,
    session_id: Optional[str] = None,
    include_mcp_tools: bool = True,
    all_tool_names: Optional[List[str]] = None,
) -> HumanInTheLoopMiddleware:
    """获取人工审批中间件（异步版本）"""
    base_config = dict(interrupt_on or DEFAULT_HIGH_RISK_TOOLS)

    if include_mcp_tools:
        mcp_tools = await get_mcp_hitl_tools_async()
        base_config.update(mcp_tools)

    config = await build_dynamic_interrupt_on_async(base_config, user, session_id, all_tool_names)

    return HumanInTheLoopMiddleware(
        interrupt_on=config,
        description_prefix=description_prefix,
    )


async def get_standard_middleware_async(
    enable_model_retry: bool = True,
    enable_tool_retry: bool = True,
    enable_summarization: bool = True,
    enable_hitl: bool = False,
    hitl_tools: Optional[Dict[str, Any]] = None,
    hitl_user=None,
    hitl_session_id: Optional[str] = None,
    hitl_all_tool_names: Optional[List[str]] = None,
    summarization_model=None,
    summarization_trigger_tokens: int = 96000,
    summarization_keep_messages: int = 4,
    model_name: str = "gpt-4o",
) -> List:
    """获取标准中间件组合（异步版本）"""
    middleware = []

    if enable_model_retry:
        middleware.append(get_model_retry_middleware())
        logger.debug("已添加 ModelRetryMiddleware")

    if enable_tool_retry:
        middleware.append(get_tool_retry_middleware())
        logger.debug("已添加 ToolRetryMiddleware")

    if enable_summarization and summarization_model is not None:
        summarization_mw = get_summarization_middleware(
            model=summarization_model,
            trigger_tokens=summarization_trigger_tokens,
            keep_messages=summarization_keep_messages,
            model_name=model_name,
        )
        if summarization_mw is not None:
            middleware.append(summarization_mw)
            logger.info("✅ 已添加 SummarizationMiddleware (trigger_tokens=%d, keep_messages=%d, model=%s)",
                        summarization_trigger_tokens, summarization_keep_messages, model_name)
        else:
            logger.warning("⚠️ SummarizationMiddleware 创建失败，返回 None")
    else:
        logger.info("⏭️ 跳过 SummarizationMiddleware: enable_summarization=%s, summarization_model=%s",
                    enable_summarization, summarization_model is not None)

    if enable_hitl:
        hitl_mw = await get_human_in_the_loop_middleware_async(
            interrupt_on=hitl_tools,
            user=hitl_user,
            session_id=hitl_session_id,
            all_tool_names=hitl_all_tool_names,
        )
        middleware.append(hitl_mw)
        logger.debug("已添加 HumanInTheLoopMiddleware（异步版本）")

    return middleware


async def get_middleware_from_config_async(
    llm_config,
    llm=None,
    agent_type: str = "standard",
    user=None,
    session_id: Optional[str] = None,
    all_tool_names: Optional[List[str]] = None,
) -> List:
    """从 LLMConfig 模型配置构建中间件列表（异步版本）"""
    if agent_type == "brain":
        return [get_model_retry_middleware(max_retries=2)]

    enable_summarization = getattr(llm_config, 'enable_summarization', True)
    enable_hitl = getattr(llm_config, 'enable_hitl', False)
    model_name = getattr(llm_config, 'name', 'gpt-4o')

    config_context_limit = getattr(llm_config, 'context_limit', None)
    if config_context_limit and config_context_limit > 0:
        context_limit = config_context_limit
    elif llm is not None:
        context_limit = get_context_limit_from_llm(llm, fallback_model_name=model_name)
    else:
        context_limit = 128000

    if agent_type == "automation":
        enable_hitl = True

    trigger_tokens = int(context_limit * 0.75)
    summarization_model = llm if enable_summarization else None

    hitl_tools = None
    if agent_type == "automation":
        hitl_tools = {
            "execute_script": {
                "allowed_decisions": ["approve", "reject"],
                "description": "自动化脚本执行需要审批",
            },
            "run_playwright": {
                "allowed_decisions": ["approve", "reject"],
                "description": "浏览器自动化操作需要审批",
            },
        }

    logger.info(
        "从 LLMConfig 构建中间件（异步）: agent_type=%s, summarization=%s, hitl=%s, context_limit=%d, trigger_tokens=%d, model=%s, user=%s, all_tools=%d",
        agent_type, enable_summarization, enable_hitl, context_limit, trigger_tokens, model_name,
        user.username if user else None, len(all_tool_names) if all_tool_names else 0
    )

    return await get_standard_middleware_async(
        enable_model_retry=True,
        enable_tool_retry=True,
        enable_summarization=enable_summarization,
        enable_hitl=enable_hitl,
        hitl_tools=hitl_tools,
        hitl_user=user,
        hitl_session_id=session_id,
        hitl_all_tool_names=all_tool_names,
        summarization_model=summarization_model,
        summarization_trigger_tokens=trigger_tokens,
        summarization_keep_messages=4,
        model_name=model_name,
    )
