from pydantic import BaseModel
from typing import Literal
import json
import uuid

from fastapi import APIRouter, HTTPException
from retrieval import retrieve
from prompts import (
    build_explain_prompt,
    build_practice_prompt,
    parse_practice_response,
)
from llm import complete, complete_structured
from prompts import PracticeQuestion, build_practice_prompt
from history import get_history, add_turn, clear_history
from observe import start_trace, log_retrieval, log_llm, finish_trace
from config import settings

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    mode: Literal["explain", "practice"] = "explain"
    session_id: str = ""        # empty = new session
    topic: str = ""             # optional topic hint for practice mode
    use_reranker: bool = True   # on by default in Phase 3


class ExplainResponse(BaseModel):
    session_id: str
    mode: str = "explain"
    response: str


class PracticeResponse(BaseModel):
    session_id: str
    mode: str = "practice"
    question: str
    choices: dict[str, str]
    correct: str
    explanation: str


@router.post("/query")
async def query(request: QueryRequest):
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Start Langfuse trace
    trace = start_trace(
        name=f"query_{request.mode}",
        input={"query": request.query, "mode": request.mode, "session_id": session_id},
    )

    try:
        # Retrieve relevant chunks
        topic_filter = request.topic or None
        if request.use_reranker:
            chunks = retrieve_reranked(request.query)
        else:
            chunks = retrieve(request.query, top_k=settings.top_k)

        log_retrieval(trace, chunks)

        # Get conversation history
        history = get_history(session_id)

        if request.mode == "explain":
            messages = build_explain_prompt(request.query, chunks, history)
            response_text = complete(messages)

            # Save turn to history
            add_turn(session_id, request.query, response_text)
            finish_trace(trace, response_text)

            return ExplainResponse(
                session_id=session_id,
                response=response_text,
            )
        elif request.mode == "practice":
            topic = request.topic or request.query
            messages = build_practice_prompt(topic, chunks, history)

            try:
                parsed = complete_structured(
                    messages=messages,
                    response_model=PracticeQuestion,
                    max_retries=3,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate valid practice question: {e}"
                )

            add_turn(session_id, request.query, parsed.model_dump_json())
            finish_trace(trace, parsed.model_dump_json())

            return PracticeResponse(
                session_id=session_id,
                question=parsed.question,
                choices=parsed.choices,
                correct=parsed.correct,
                explanation=parsed.explanation,
            )

    except HTTPException:
        raise
    except Exception as e:
        finish_trace(trace, f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
