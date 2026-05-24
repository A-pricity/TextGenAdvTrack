from typing import Literal

from pydantic import BaseModel, Field, model_validator

EvasionLanguage = Literal["zh", "en", "ru", "unknown"]


class DetectionRow(BaseModel):
    sample_id: str
    split: Literal["train", "dev", "rewrite_dev"]
    language: Literal["zh", "en", "ru"]
    domain: str
    label: Literal[0, 1]
    text_type: Literal["human", "ai_original", "ai_rewritten"]
    source_name: str
    source_model: str | None = None
    prompt_type: str | None = None
    prompt_id: str | None = None
    rewrite_type: str | None = None
    parent_id: str | None = None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rewrite_fields(self):
        if self.text_type == "ai_rewritten":
            if not self.parent_id:
                raise ValueError("ai_rewritten rows require parent_id")
            if not self.rewrite_type:
                raise ValueError("ai_rewritten rows require rewrite_type")
        return self


class EvasionCandidateRow(BaseModel):
    candidate_id: str
    parent_id: str
    language: EvasionLanguage
    domain: str
    source_model: str
    prompt_type: str
    rewrite_model: str
    rewrite_type: str
    semantic_score: float
    proxy_score_1: float
    proxy_score_2: float
    selected: bool
    text: str = Field(min_length=1)


class EvasionSourceRow(BaseModel):
    sample_id: str
    language: EvasionLanguage
    domain: str
    source_model: str
    prompt_type: str
    prompt_id: str
    prompt: str | None = None
    source_text: str = Field(min_length=1)


class DetectionSubmitInputRow(BaseModel):
    prompt: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvasionOfficialInputRow(BaseModel):
    prompt: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal[0, 1]


class EvasionSelectedRow(BaseModel):
    sample_id: str
    parent_id: str
    language: EvasionLanguage
    final_text: str = Field(min_length=1)
    proxy_score_1: float
    proxy_score_2: float
    selection_reason: str = Field(min_length=1)
