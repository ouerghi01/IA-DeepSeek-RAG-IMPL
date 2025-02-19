
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
from langgraph.checkpoint.memory import MemorySaver
from tools import *
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.agents import AgentAction
from typing import List, Optional, Sequence, Union
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.tools.render import ToolsRenderer, render_text_description
from langchain.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

from langchain.agents import AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
# Define or import the tools_renderer function
def tools_renderer(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

memory = MemorySaver()
model_name = "deepseek-r1:1.5b"

llm = ChatOllama(model=model_name, base_url="http://localhost:11434")
session, db = initialize_database_session()
toolkit = CassandraDatabaseToolkit(db=db)
tools = toolkit.get_tools()

# Generate tool names from the tools

# User query
user_query = "What are the names of the tables in the keyspace 'store_key'?"

# Updated template with placeholders for tools, tool_names, and agent_scratchpad
template = """You are an intelligent AI assistant with access to various tools that help you answer questions. 
You must always use the tools before arriving at a final answer.

You have access to the following tools:

{tools}

### **Instructions:**
1. Carefully analyze the question.
2. Decide the best tool to use and what input it requires.
3. Execute the tool and observe the result.
4. If necessary, use additional tools or refine your queries.
5. Once confident, provide the final answer.

### **Strict Format to Follow:**
- **Question:** the input question you must answer.
- **Thought:** analyze what to do next.
- **Action:** specify the tool to use, should be one of [{tool_names}].
- **Action Input:** provide the input for the chosen tool.
- **Observation:** record the tool’s response.
- **(Repeat the above steps as needed)**
- **Final Thought:** confirm that you have all the required information.
- **Final Answer:** provide the final response to the user.

---

**Example Interaction:**

Question: What are the names of the tables in the keyspace 'store_key'?

Thought: I need to retrieve the table names in the keyspace 'store_key'. The tool `cassandra_db_schema` can help with this.
Action: cassandra_db_schema
Action Input: "DESCRIBE TABLES store_key"
Observation: ['customers', 'orders', 'products']

Thought: I now have the final answer.
Final Answer: The tables in the keyspace 'store_key' are: customers, orders, and products.

---

### **Your Turn! Follow the format strictly.**
Question: {input}
Thought:{agent_scratchpad}"""


# Create the prompt template
prompt = PromptTemplate.from_template(template)


# Check for any missing variables in the template
MessageFormatter = Callable[[Sequence[Tuple[AgentAction, str]]], List[BaseMessage]]
# Prepare input for the agent
input_query = ("QUERY_PATH_PROMPT +\n\nHere is your task: Find all the questions in the response_table  has uploaded to the   'store_key' keyspace .")
def create_tool_calling_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    *,
    message_formatter: MessageFormatter = format_to_tool_messages,
) -> Runnable:
    """Create an agent that uses tools.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
        message_formatter: Formatter function to convert (AgentAction, tool output)
            tuples into FunctionMessages.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            model = ChatAnthropic(model="claude-3-opus-20240229")

            @tool
            def magic_function(input: int) -> int:
                \"\"\"Applies a magic function to an input.\"\"\"
                return input + 2

            tools = [magic_function]

            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if not hasattr(llm, "bind_tools"):
        raise ValueError(
            "This function requires a .bind_tools method be implemented on the LLM.",
        )
    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: message_formatter(x["intermediate_steps"])
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return agent

# Run the agent and print the response
def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: Optional[AgentOutputParser] = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    """Create an agent that uses ReAct prompting.

    Based on paper "ReAct: Synergizing Reasoning and Acting in Language Models"
    (https://arxiv.org/abs/2210.03629)

    .. warning::
       This implementation is based on the foundational ReAct paper but is older and not well-suited for production applications.
       For a more robust and feature-rich implementation, we recommend using the `create_react_agent` function from the LangGraph library.
       See the [reference doc](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)
       for more information.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.
        output_parser: AgentOutputParser for parse the LLM output.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM. Default is `render_text_description`.
        stop_sequence: bool or list of str.
            If True, adds a stop token of "Observation:" to avoid hallucinates.
            If False, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Examples:

        .. code-block:: python

            from langchain import hub
            from langchain_community.llms import OpenAI
            from langchain.agents import AgentExecutor, create_react_agent

            prompt = hub.pull("hwchase17/react")
            model = OpenAI()
            tools = ...

            agent = create_react_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Use with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    # Notice that chat_history is a string
                    # since this prompt is aimed at LLMs, not chat models
                    "chat_history": "Human: My name is Bob\\nAI: Hello Bob!",
                }
            )

    Prompt:

        The prompt must have input keys:
            * `tools`: contains descriptions and arguments for each tool.
            * `tool_names`: contains all tool names.
            * `agent_scratchpad`: contains previous agent actions and tool outputs as a string.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate

            template = '''Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}'''

            prompt = PromptTemplate.from_template(template)
    """  # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm
    output_parser = output_parser or ReActSingleInputOutputParser()
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )
    return agent
agent_ex=create_tool_calling_agent(llm,tools,prompt)

agent_executor = AgentExecutor(agent=agent_ex, tools=tools, verbose=True)
response = agent_executor.invoke({"input": input_query})