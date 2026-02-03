"""Pytest Testing Patterns for FastAPI.

Keywords: test, pytest, testing, httpx, async, fixture, mock

This pattern shows how to test FastAPI applications properly.

Requirements:
    pip install pytest pytest-asyncio httpx

Key points:
- Use httpx.AsyncClient for async tests
- Use fixtures for setup/teardown
- Test both success and error cases
- Mock external dependencies
"""

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

# ============================================================================
# Example Application to Test
# ============================================================================

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float


# Fake database
fake_db: dict[int, Item] = {}


async def get_db() -> dict:
    """Database dependency."""
    return fake_db


@app.post("/items", response_model=Item)
async def create_item(item: Item, db: dict = Depends(get_db)):
    item_id = len(db) + 1
    db[item_id] = item
    return item


@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int, db: dict = Depends(get_db)):
    if item_id not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    return db[item_id]


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def anyio_backend():
    """Use asyncio for async tests."""
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client fixture.

    Use this for all async endpoint tests.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test."""
    fake_db.clear()
    yield
    fake_db.clear()


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.anyio
async def test_create_item(client: AsyncClient):
    """Test successful item creation."""
    response = await client.post(
        "/items",
        json={"name": "Test Item", "price": 9.99}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Item"
    assert data["price"] == 9.99


@pytest.mark.anyio
async def test_get_item(client: AsyncClient):
    """Test getting an existing item."""
    # First create an item
    await client.post("/items", json={"name": "Test", "price": 1.0})

    # Then get it
    response = await client.get("/items/1")

    assert response.status_code == 200
    assert response.json()["name"] == "Test"


@pytest.mark.anyio
async def test_get_item_not_found(client: AsyncClient):
    """Test 404 when item doesn't exist."""
    response = await client.get("/items/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"


@pytest.mark.anyio
async def test_create_item_validation_error(client: AsyncClient):
    """Test validation error for invalid data."""
    response = await client.post(
        "/items",
        json={"name": "Test"}  # Missing required 'price' field
    )

    assert response.status_code == 422  # Validation error


# ============================================================================
# Testing with Mocks
# ============================================================================

@pytest.mark.anyio
async def test_with_mocked_dependency(client: AsyncClient):
    """Test with mocked database dependency."""
    mock_db = {1: Item(name="Mocked", price=99.99)}

    # Override dependency
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        response = await client.get("/items/1")
        assert response.status_code == 200
        assert response.json()["name"] == "Mocked"
    finally:
        # Clean up override
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_with_patched_external_service():
    """Test with patched external service call."""
    # Example: patching an external API call
    with patch("module.external_api.fetch_data", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"data": "mocked"}

        # Your test code here
        result = await mock_fetch()
        assert result == {"data": "mocked"}


# ============================================================================
# Parameterized Tests
# ============================================================================

@pytest.mark.anyio
@pytest.mark.parametrize("name,price,expected_status", [
    ("Valid Item", 10.0, 200),
    ("Another Item", 0.01, 200),
    ("Free Item", 0.0, 200),
])
async def test_create_item_parametrized(
    client: AsyncClient,
    name: str,
    price: float,
    expected_status: int
):
    """Parameterized test for item creation."""
    response = await client.post(
        "/items",
        json={"name": name, "price": price}
    )
    assert response.status_code == expected_status


# ============================================================================
# Test Utilities
# ============================================================================

class TestHelpers:
    """Helper methods for tests."""

    @staticmethod
    async def create_test_item(client: AsyncClient, name: str = "Test", price: float = 1.0) -> dict:
        """Create a test item and return the response data."""
        response = await client.post("/items", json={"name": name, "price": price})
        return response.json()

    @staticmethod
    async def assert_item_exists(client: AsyncClient, item_id: int):
        """Assert that an item exists."""
        response = await client.get(f"/items/{item_id}")
        assert response.status_code == 200


# Usage in tests:
@pytest.mark.anyio
async def test_with_helpers(client: AsyncClient):
    """Test using helper methods."""
    item = await TestHelpers.create_test_item(client, "Helper Item", 5.0)
    assert item["name"] == "Helper Item"

    await TestHelpers.assert_item_exists(client, 1)
