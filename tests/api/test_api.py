def test_ask_endpoint_success(client, mock_rag_pipeline):
    response = client.post(
        "/ask",
        json={"question": "What is SAP?"}
    )

    data = response.json()

    assert response.status_code == 200

    assert "answer" in data
    assert "sources" in data

    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)


def test_ask_endpoint_missing_question(client):
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_endpoint_rag_failure(client, mock_rag_pipeline):
    mock_rag_pipeline.ask.side_effect = RuntimeError("Vector store down")

    response = client.post(
        "/ask",
        json={"question": "test"}
    )

    assert response.status_code == 500


def test_ask_endpoint_no_documents(client, mock_rag_pipeline):
    mock_rag_pipeline.ask.return_value = {
        "answer": "I don't know based on the provided documents.",
        "sources": []
    }

    response = client.post(
        "/ask",
        json={"question": "unknown topic"}
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []