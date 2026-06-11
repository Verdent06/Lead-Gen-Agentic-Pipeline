import httpx, asyncio

from src.config import Config

async def test():
    print(f"Key being used: '{Config.HUNTER_API_KEY}'")
    r = await httpx.AsyncClient().get(
        "https://api.hunter.io/v2/domain-search",
        params={"domain": "atb.com", "limit": 10, "department": "hr"},
        headers={"Authorization": f"Bearer {Config.HUNTER_API_KEY}"}
    )
    print(r.json())

asyncio.run(test())