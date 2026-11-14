import asyncio
from mcp_client import client


async def main():

    tools = await client.get_tools(
        server_name="aviationstack"
    )

    print("\nAVAILABLE AVIATIONSTACK TOOLS\n")

    for tool in tools:
        print("\n" + "=" * 60)
        print("Tool:", tool.name)
        print("Description:", tool.description)

        schema = tool.args_schema

        print("\nArguments:")

        if isinstance(schema, dict):
            print(schema)
        elif hasattr(schema, "model_json_schema"):
            print(schema.model_json_schema())
        else:
            print(schema)

        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())