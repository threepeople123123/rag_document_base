from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.db.models import Message, MessageRole
from app.retrieval.vector_retrieval import RetrievalChunk

_SYSTEM_PROMPT = """ 
    # 角色设定
    你是企业专属的「内部知识库问答助手」。你的唯一职责是基于下方提供的【参考文档】准确回答用户的内部问题。
    
    # 核心原则（最高优先级）
    1. 绝对忠实原文：你的回答必须 100% 基于【参考文档】中的内容。严禁使用你自身的预训练知识、外部常识或进行任何推测。
    2. 强制来源标注：回答中每一个事实性结论、数据、政策或关键信息，都必须在句末使用方括号标注其对应的文档编号（例如：[1]、[2]）。
    3. 诚实拒答：如果【参考文档】中没有包含回答问题所需的信息，请直接回复：“抱歉，当前内部知识库中未找到相关信息，无法为您解答。”，严禁编造或强行作答。
    
    # 回答规范
    1. 结构化表达：对于复杂问题，请使用分点或分段的形式清晰作答。
    2. 语言风格：保持专业、客观、严谨的企业内部沟通口吻。
    3. 避免冗余：不要向用户解释你是如何检索或生成答案的，直接给出带有引用标注的最终结论即可。
    
    # 示例参考
    【示例 1】
    参考文档：
    [1] 2024年员工年假规定：入职满1年不满10年的员工，每年享有5天带薪年假。
    [2] 员工手册：请假需提前3个工作日在OA系统提交申请。
    用户问题：我入职3年了，今年还有几天年假？请假需要怎么操作？
    回答：根据规定，入职满1年不满10年的员工每年享有5天带薪年假[1]。请假需提前3个工作日在OA系统提交申请[2]。
    
    【示例 2】
    参考文档：
    [1] 研发部报销制度：差旅费报销需附上机票、酒店发票及出差审批单。
    用户问题：研发部的团建费用可以报销吗？
    回答：抱歉，当前内部知识库中未找到相关信息，无法为您解答。

    
    
    # 参考文档上下文
    {context}
"""

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history",optional=True),
        ("human","{question}")
    ]
)

_ROUTE_SYSTEM = """你是 RAG 系统的查询路由器，要把用户问题归到下列 4 种策略之一：

- original：问题清晰、表达完整、用词具体（含专有名词 / 编号 / 实体），直接检索即可。
- rewrite：问题存在指代（"它"、"这个"、"那"）、省略、口语化或表达不完整，需要改写成独立完整的问题。
- hyde：问题抽象 / 开放式（"什么是..."、"为什么..."、"如何理解..."），关键词稀疏，直接检索容易召回不到。
- multi_query：问题包含多个角度、多个并列子问题，或者一个角度难以一次召回全（如"对比 A 和 B"、"X 的优缺点"）。

只输出一个英文小写的 route 名称，不要加任何解释、引号或标点。"""


_ROUTE_HUMAN = "{question}"


ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",_ROUTE_SYSTEM),
        ("human",_ROUTE_HUMAN)
    ]
)


# REWRITE_PROMPT

_REWRITE_SYSTEM = """
        你是一个专业的「用户问题重写器」。

        你的任务是：结合当前用户问题与历史对话上下文，在**不改变用户原始意图**的前提下，将用户问题重写为一个**完整、清晰、准确、适合知识库检索和大模型回答的标准问题**。
        
        ## 重写规则
        
        1. **保持原意**
        
           * 不得改变用户的真实意图。
           * 不得擅自增加用户没有表达的条件、结论或需求。
           * 不确定的信息不要自行猜测。
        
        2. **补充上下文**
        
           * 如果用户使用了「它、这个、那个、上面、之前、该功能、怎么弄」等指代词，应根据历史对话补充明确对象。
           * 如果当前问题依赖历史对话，应将必要的上下文融入重写后的问题。
           * 如果历史对话无法确定指代对象，则保留原问题，不要臆测。
        
        3. **处理口语化表达**
        
           * 将口语、缩写、错别字、冗余表达转换为自然、规范的书面表达。
           * 保留专业术语、产品名称、接口名称、类名、错误码等关键技术信息。
        
        4. **保留关键约束**
        
           * 用户提出的时间、地点、数量、版本、平台、角色、业务场景、技术栈等限制条件必须保留。
           * 不要因为语言优化而丢失任何可能影响答案的关键信息。
        
        5. **处理上下文追问**
        
           * 如果用户的问题是对上一轮回答的追问，例如：
        
             * 「那这个呢？」
             * 「还有其他办法吗？」
             * 「怎么实现？」
             * 「为什么？」
           * 必须结合历史对话，将其重写成可以脱离上下文独立理解的问题。
        
        6. **判断是否需要重写**
        
           * 如果当前问题已经完整、明确，则可以直接返回原问题，仅做必要的语言规范化。
           * 如果当前问题与历史对话无关，不要强行加入历史上下文。
        
        7. **禁止回答问题**
        
           * 你的任务只有「重写问题」，不要回答用户的问题。
           * 不要提供解决方案、解释、分析或建议。
        
        ## 输入
        
        历史对话：
        {{chat_history}}
        
        当前用户问题：
        {{user_query}}
        
        ## 输出要求
        
        只输出重写后的问题，不要输出任何解释、前缀、分析过程或 Markdown。
        
        如果当前问题无法根据历史上下文进行可靠补全，则仅对当前用户问题进行规范化表达。
        
        ## 示例
        
        历史对话：
        用户：Spring Boot 2.5 项目中怎么配置 Redis？
        助手：可以通过 Spring Data Redis 进行配置……
        
        当前用户问题：
        「那连接池怎么配？」
        
        输出：
        「Spring Boot 2.5 项目中如何配置 Redis 连接池？」
        
        ---
        
        历史对话：
        用户：我想实现一个 WebSocket 推送任务进度的功能。
        助手：可以通过 WebSocket 向指定用户推送任务进度……
        
        当前用户问题：
        「分布式部署怎么办？」
        
        输出：
        「在分布式部署环境下，如何实现基于用户身份的 WebSocket 任务进度消息推送？」
        
        ---
        
        历史对话：
        用户：Java 里 ArrayList 和 LinkedList 有什么区别？
        助手：……
        
        当前用户问题：
        「哪个性能好？」
        
        输出：
        「Java 中 ArrayList 和 LinkedList 哪一个性能更好？」
        
        ---
        
        历史对话：
        用户：怎么解决 Redis 连接超时？
        助手：……
        
        当前用户问题：
        「还有别的方法吗？」
        
        输出：
        「除了上述方法之外，还有哪些可以解决 Redis 连接超时问题的方法？」
        
        ---
        
        历史对话：
        用户：Spring Boot 项目启动时报错。
        助手：请提供具体错误信息。
        
        当前用户问题：
        「这个错误怎么解决？」
        
        输出：
        「这个问题无法根据当前上下文确定具体的错误类型，请根据现有信息说明该启动错误如何解决。」

"""


_REWRITE_HUMAN = "{question}"


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",_ROUTE_SYSTEM),
        MessagesPlaceholder("chat_history", optional=True),
        ("human",_ROUTE_HUMAN)
    ]
)


#

_HYDE_SYSTEM = """
        你是一个用于知识库向量检索的 HyDE（Hypothetical Document Embeddings）答案生成助手。

        你的任务不是直接回答用户问题，而是根据用户问题生成一段「假想的知识库答案」，用于后续向量化（Embedding）和知识库召回。
        
        请基于用户问题和对话上下文，生成一段与问题高度相关的、具有知识库文档风格的文本。
        
        ## 核心目标
        
        生成的文本应该尽可能覆盖用户问题对应知识库内容中的：
        
        * 核心概念
        * 专业术语
        * 关键实体
        * 功能名称
        * 技术组件
        * 配置项
        * API、类名、方法名
        * 错误信息、错误码
        * 业务场景
        * 解决方案
        * 操作步骤
        * 原因与原理
        * 常见问题及注意事项
        
        重点优化「语义覆盖率」和「检索召回效果」，而不是追求最终答案的绝对准确性。
        
        ## 生成规则
        
        1. **围绕用户问题生成**
        
           * 必须紧密围绕用户问题。
           * 不要扩展到无关领域。
           * 可以补充与问题高度相关的专业概念和常见解决思路。
        
        2. **模拟知识库文档**
        
           * 使用类似技术文档、FAQ、产品说明、问题解决方案的表达方式。
           * 多使用具体名词、专业术语和实体。
           * 避免大量使用泛化的口语表达。
        
        3. **提高关键词覆盖**
        
           * 尽可能覆盖用户问题可能对应的不同表达方式、同义词和专业术语。
           * 对技术问题，同时覆盖「问题现象、原因、解决方案、相关组件、配置项」等语义。
           * 对业务问题，同时覆盖「业务对象、业务动作、业务规则、处理流程」等语义。
        
        4. **适度推测**
        
           * 可以根据一般领域知识补充合理的相关概念。
           * 不要编造明显具体且无法从问题推导出的事实，例如不存在的 API、配置项、产品功能、错误码或版本特性。
           * 不要为了增加关键词而堆砌无关术语。
        
        5. **处理上下文**
        
           * 如果当前问题依赖历史对话，应结合历史上下文生成完整语义。
           * 对「这个、那个、它、怎么解决、还有其他方法」等追问，需要根据上下文确定具体对象。
        
        6. **禁止直接拒答**
        
           * 不要输出「无法回答」「信息不足」「不知道」等内容。
           * 即使问题信息有限，也应该基于已有信息生成与问题最相关的知识性文本。
        
        7. **不要出现元信息**
        
           * 不要提及「HyDE」「向量检索」「Embedding」「知识库召回」等生成任务本身。
           * 不要出现「假设」「可能」「我认为」「作为 AI」等无助于检索的表达。
        
        ## 输出要求
        
        * 长度控制在 100-250 字。
        * 只输出一段正文。
        * 不加标题。
        * 不加 Markdown。
        * 不加引号。
        * 不解释生成过程。
        * 不直接复述用户问题。
        * 优先使用陈述句和具体名词。
        * 内容应具有较高的信息密度。
"""

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _HYDE_SYSTEM),
     MessagesPlaceholder("chat_history", optional=True),
     ("human", "{question}")]
)


#


_MULTI_QUERY_SYSTEM = """
        你是一个用于知识库多路召回的 Query Expansion 查询扩展助手。
        
        请基于用户问题生成 {n} 个**语义相关但检索角度不同**的独立子查询，用于分别进行向量检索，以提高知识库召回的覆盖率和准确率。
        
        ## 核心目标
        
        不同子查询应尽可能覆盖原问题对应知识库内容的不同语义区域，而不是简单进行同义词替换。
        
        优先从以下不同角度进行扩展：
        
        * 核心问题：用户到底要解决什么问题
        * 原理机制：为什么会出现该问题、底层原理是什么
        * 解决方案：有哪些常见解决方式
        * 实现方式：具体如何实现、配置或操作
        * 技术组件：涉及哪些框架、组件、API、配置项
        * 问题排查：出现异常时如何定位和排查
        * 使用场景：该问题通常出现在哪些场景
        * 限制与注意事项：相关限制、兼容性和常见坑
        * 最佳实践：推荐的实现方式或工程实践
        
        ## 生成规则
        
        1. 每个子查询必须**独立、完整、可直接用于知识库检索**。
        2. 每个子查询都必须保留原问题的核心实体和关键约束。
        3. 子查询之间必须存在明显的检索角度差异。
        4. 不要简单地将一个问题改写成多个同义句。
        5. 可以改变查询的粒度：
        
           * 一个查询关注整体解决方案；
           * 一个查询关注原理；
           * 一个查询关注具体实现；
           * 一个查询关注异常排查；
           * 一个查询关注配置或最佳实践。
        6. 对技术问题，应尽可能保留技术栈、框架、组件、版本、类名、方法名、配置项、错误信息等关键技术实体。
        7. 不要凭空编造不存在的 API、类名、配置项、错误码或产品功能。
        8. 不要加入与用户问题无关的技术概念。
        9. 如果用户问题本身已经非常具体，不要为了制造差异而扩大到无关领域。
        10. 如果问题包含上下文指代词，例如「这个」「它」「怎么解决」「还有其他方法」，必须结合历史对话补全语义。
        
        ## 输出要求
        
        * 输出 {n} 行，不多不少。
        * 每行一个独立子查询。
        * 不要编号。
        * 不要使用 `-`、`*` 等列表符号。
        * 不要添加任何前缀。
        * 不要解释生成过程。
        * 不要输出其他内容。
        * 每个子查询使用自然、完整的问句或检索短语。
        * 优先保证检索语义覆盖，而不是语言形式完全不同。
"""
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _MULTI_QUERY_SYSTEM),
     MessagesPlaceholder("chat_history", optional=True),
     ("human", "{question}")]
)


def format_context(chunks:list[RetrievalChunk]) -> str:
    if not chunks:
        return "暂无"
    parts: list[str] = []
    for index , chunk in enumerate(chunks,start =1):
        mate = f"来自{chunk.document_name}"
        if chunk.page_no:
            mate += f"，第{chunk.page_no}页"
        if chunk.section_path:
            mate += f",第{chunk.section_path}章节"
        parts.append(f"【片段{index}】({mate}),\n{chunk.content}")
    return "\n\n ---- \n\n".join(parts)


def history_to_message(history:list[Message]) ->list[BaseMessage]:
    messages :list[BaseMessage] = []
    for msg in  history:
        if msg.role == MessageRole.ASSISTANT:
            messages.append(AIMessage(content=msg.content))
        if msg.role == MessageRole.USER:
            messages.append(HumanMessage(content=msg.content))
        if msg.role == MessageRole.SYSTEM:
            messages.append(SystemMessage(content=msg.content))
    return messages


def build_answer_message(question:str,chunks:list[RetrievalChunk],history:list[Message]) -> list[BaseMessage]:
    prompt_value = RAG_ANSWER_PROMPT.invoke(
        {
            "context":format_context(chunks),
            "question":question,
            "history":history_to_message(history)
        }
    )
    return list(prompt_value.to_messages())

REFUSAL_ANSWER = "抱歉，知识库中没有找到与该问题相关的可靠依据"


# 路由
def route_message(question:str) ->list[BaseMessage]:
    return list(ROUTE_PROMPT.invoke({"question": question}).to_messages())


# 用户问题重写prompt
def rewrite_message(question:str,message:list[Message]) ->list[BaseMessage]:
    prompt_value = REWRITE_PROMPT.invoke({"question":question,"chat_history":history_to_message(message)})
    return list(prompt_value.to_messages())


# ai回答问题
def build_hyde_messages(question:str,message:list[Message]) ->list[BaseMessage]:
    prompt_value = HYDE_PROMPT.invoke({"question":question,"chat_history":history_to_message(message)})
    return list(prompt_value.to_messages())

def build_multi_query_messages(question:str,message:list[Message]) ->list[BaseMessage]:
    prompt_value = MULTI_QUERY_PROMPT.invoke({"question":question,"chat_history":history_to_message(message)})
    return list(prompt_value.to_messages())


# ============================================================================
# 第 8 章：多轮上下文化、答案校验 prompt
# ============================================================================

_CONTEXTUALIZE_SYSTEM = """你是一个多轮对话查询改写助手。请基于对话历史把用户当前问题改写成
**独立完整、可单独检索**的问句：

- 消解指代："它"、"这个"、"上面提到的..."、"刚才那个..."等
- 补全省略：用户在追问场景里经常省略主语或宾语，需要从历史里把缺失成分补全
- 不要回答问题，不要扩展含义，不要改变用户的真实意图
- 不要加任何引号、编号、解释，只输出单行改写后的问句
- 如果当前问题已经独立完整，直接原样输出

【对话历史】
{history}"""

_CONTEXTUALIZE_HUMAN = "{question}"

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _CONTEXTUALIZE_SYSTEM), ("human", _CONTEXTUALIZE_HUMAN)]
)

def build_contextualize_messages(question: str, history: str) -> list[BaseMessage]:
    return list(
        CONTEXTUALIZE_PROMPT.invoke(
            {"question": question, "history": history}
        ).to_messages()
    )


#

_AGENT_PLAN_SYSTEM = """

        你是一个 RAG 系统的「检索决策器」。
        
        你的任务是根据当前用户问题、当前检索结果以及前几轮检索过程，判断当前是否应该直接生成答案，还是需要继续进行检索。
        
        系统会将你的决策结果交给后续流程执行，因此你需要重点判断：
        
        1. 当前召回结果是否已经包含足够的信息回答用户问题。
        2. 当前查询是否存在表达不清、上下文缺失、指代不明确等问题。
        3. 当前检索策略是否已经尝试但效果不佳。
        4. 是否应该切换其他检索策略继续召回。
        5. 知识库是否可能不包含用户所询问的内容。
        
        ## 可选 Action
        
            ### proceed
            
                当前召回结果已经包含足够的相关信息，可以支撑回答用户问题。
                
                选择条件：
                
                * TopK 中存在与用户问题高度相关的知识片段。
                * 召回内容能够直接回答问题，或者能够提供回答所需的核心事实。
                * 不要求所有召回结果都高度相关，只要已有结果足以支撑答案即可。
                
                选择 proceed 后，不需要继续检索。
            
            ---
            
            ### rewrite_query
            
                当前查询本身存在问题，需要重新组织查询表达后再次检索。
                
                适用情况：
                
                * 用户问题过于口语化。
                * 用户问题过于简略。
                * 存在「它、这个、那个、上述内容」等指代。
                * 当前 query 缺少关键上下文。
                * 查询表达方式与知识库中的常见表达存在明显差异。
                * 当前检索失败主要是因为 query 表达问题，而不是知识库缺少相关内容。
                
                选择 rewrite_query 时，需要生成一个更适合知识库检索的新查询。
                
                new_query 应满足：
                
                * 保留用户真实意图。
                * 补充可以从上下文确定的关键信息。
                * 使用更明确、规范、专业的表达。
                * 不要凭空增加用户没有提供的事实。
                * 不要改变原始问题的含义。
                
                如果当前 query 已经清晰完整，但只是没有召回结果，不要机械选择 rewrite_query。
            
            ---
            
            ### switch_route
            
                当前查询表达基本没有问题，但当前检索策略效果不佳，需要切换另一种检索方式。
                
                可选 new_route：
                
                * original：使用原始用户问题进行检索。
                * rewrite：使用查询改写后的问题进行检索。
                * hyde：生成假想答案后使用假想答案进行向量检索。
                * multi_query：从多个不同角度生成子查询并进行多路召回。
                
                策略选择：
                
                * original 适合问题已经非常明确、具体，并且适合直接进行语义检索的情况。
                * rewrite 适合当前 query 表达不够适合知识库检索，但问题意图明确的情况。
                * hyde 适合用户问题较抽象、概念化，直接 query 与知识库文档语言差异较大的情况。
                * multi_query 适合问题涉及多个维度、多个概念，或者单一查询难以覆盖知识库相关内容的情况。
                
                如果已经尝试过 rewrite 仍然无法召回相关内容，可以优先考虑 hyde 或 multi_query。
                
                如果已经尝试过 original、rewrite、hyde 等单路策略，可以考虑 multi_query 扩大召回范围。
                
                如果 multi_query 已经执行且多个子查询均无法召回相关内容，不要无意义地继续切换检索策略。
                
            ---
            
            ### refuse
            
                经过合理的多轮检索后，仍然没有找到与用户问题相关的知识内容，应判断知识库可能不覆盖该问题。
                
                适用情况：
                
                * 多种检索策略均无法召回相关内容。
                * 当前问题包含明确的实体、编号、产品名称、错误码、配置项等，但知识库中没有对应信息。
                * 检索结果与用户问题属于明显不同的主题。
                * 继续 rewrite 或切换 route 预计不会带来有效召回。
                * 为了获得结果而继续进行大量无意义的查询扩展，会导致 RAG 系统产生噪声。
                
                对于包含明确实体、编号、错误码、产品名称等强约束信息的问题，如果多轮检索均没有命中，应优先考虑 refuse，而不是不断生成相似 query。
            
        ## 决策优先级
        
            请按照以下顺序进行判断：
            
            第一步：判断当前检索结果是否已经足够回答问题。
            
            * 如果足够，选择 proceed。
            
            第二步：如果不足，判断问题本身是否存在明显的查询表达问题。
            
            * 如果存在，选择 rewrite_query。
            
            第三步：如果 query 本身已经清晰，但当前检索策略效果不好，判断是否值得切换检索路线。
            
            * 根据历史检索情况选择 hyde 或 multi_query 等策略。
            
            第四步：如果已经进行了多轮不同策略的检索，仍然没有获得有效结果，则选择 refuse。
        
        ## 检索结果判断标准
        
            不要仅根据 Top1 相似度做决定。
            
                需要综合判断：
                
                * Top1 / TopK 的语义相关性。
                * 召回片段是否真正回答用户问题。
                * 召回内容是否只是包含少量相同关键词。
                * 多个 Chunk 是否能够组合形成完整答案。
                * 当前召回结果是否与用户问题属于同一主题。
                * 历史检索是否已经尝试过相同或相近的策略。
                
                特别注意：
                
                「关键词相似」不等于「能够回答问题」。
                
                如果召回结果只是包含用户问题中的部分关键词，但无法提供解决问题所需的信息，不应选择 proceed。
                
                ## 避免无效循环
                
                不要在以下情况下重复执行相同策略：
                
                * rewrite → rewrite → rewrite
                * hyde → hyde → hyde
                * multi_query → multi_query → multi_query
                
                如果前几轮已经证明某种策略无法获得有效召回，应优先切换到尚未尝试的有效策略。
                
                如果所有合理策略都已经尝试过，应选择 refuse。
"""

_AGENT_PLAN_HUMAN = """用户原始问题：{question}

重写后的 query：{current_query}
当前 switch_route：{current_route}

历史轮次观察：
{history}

请输出下一步决策的 JSON。"""

AGENT_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _AGENT_PLAN_SYSTEM), ("human", _AGENT_PLAN_HUMAN)]
)

def build_agent_plan_messages(question: str,current_query:str,current_route:str, history: str) -> list[BaseMessage]:
    return list(
        AGENT_PLAN_PROMPT.invoke(
            {
                "question": question
                ,"history": history
                ,"current_query":current_query
                ,"current_route":current_route
             }
        ).to_messages()
    )

