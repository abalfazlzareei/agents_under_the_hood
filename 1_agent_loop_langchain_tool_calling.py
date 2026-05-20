from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10


MODEl = "gpt-oss:20b"

@tool
def get_product_price(product:str)-> float:
    "Look up the price of a product in the catalogue"
    print(f".    >> Executing get_product_price (product={product})")
    prices = {"laptob":30, "keyboard":10, "pen":15}
    return prices.get(product,0)

@tool
def apply_discount(price:float, discount_tier:str)->float:
    """Apply a discount to the price and return the final price
        Available tiers: bronze, silver, gold. 
    """
    print(f".  >> Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze":5, "silver":10, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount/100),2)

# ----- Agent Loop ----
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
        tools = [get_product_price, apply_discount]
        tools_dict = {t.name: t for t in tools}
        print(f"tools_dict:{tools_dict}")

        llm = init_chat_model(f"ollama:gpt-oss:20b",temperature = 0)
        llm_with_tools = llm.bind_tools(tools)
        print("Question: ", question)
        print("="*60)
        Messages = [
              SystemMessage(
                 content=(   "You are a helpful shopping assistant"
                    "you have access to a product catalog tool"
                    "and a discount tool.\n\n"
                    "STRICT RULES - you must follow these exactly:\n"
                    "1. NEVER guess or assume any product price."
                    "You must call get_product_price first to get the real price.\n"
                    "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price"
                    "returned by get_product_price - do NOT pass a made-up number.\n"
                    "3. NEVER calculate discounts yourself using math."
                    "Always use the apply_discount tool.\n"
                    "4. If the user does not specify a discount tier,"
                    "ask them which tier to use - do NOT assume one."
                    )
              ),
              HumanMessage(content = question)
        ]

        for iteration in range(1,MAX_ITERATIONS + 1):
            print(f"\n -------Iteration {iteration} ---")
            ai_message = llm_with_tools.invoke(Messages)
            print("AI_Message:", ai_message)
            print("AI Message:", ai_message.content)
            
            tool_calls = ai_message.tool_calls
            print("Tool Calls:", ai_message.tool_calls)
            if not tool_calls:
                print(f"\n Final Answer",ai_message.content)
                return ai_message.content

            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"tool {tool_name} Not Found!")
            
            observation = tool_to_use.invoke(tool_args)
            print(f"Tool Result: {observation}")

            Messages.append(ai_message)
            Messages.append(
                 ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )

        print("ERROR: Max Iterations reached without a final answer.")
        return None

if __name__ == "__main__":
        print("Hello Langchain Agent (.bind_tools)!")
        print()
        print("what is the price of a laptob after applying a gold discount?")
        run_agent("what is the price of a laptob after applying a gold discount?")







