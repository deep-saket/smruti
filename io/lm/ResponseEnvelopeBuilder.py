# modules/processing/ResponseEnvelopeBuilder.py

from typing import Optional
from common.CallableComponent import CallableComponent
from io.lm.ChatResponse import ChatResponse
from io.lm.ResponseEnvelope import ResponseEnvelope, Diagnostics, ModelInfo, GenerationParams, SafetyFlags, LengthMetrics, FallbackInfo
class ResponseEnvelopeBuilder(CallableComponent):
    """
    Combines a ChatResponse and computed diagnostics & metadata
    into a full ResponseEnvelope.
    """
    def __call__(
        self,
        response: ChatResponse,
        diagnostics: Diagnostics,
        model_info: ModelInfo,
        generation_params: GenerationParams,
        safety: Optional[SafetyFlags] = None,
        length_metrics: Optional[LengthMetrics] = None,
        fallback: Optional[FallbackInfo] = None
    ) -> ResponseEnvelope:
        return ResponseEnvelope(
            model_info=model_info,
            generation_params=generation_params,
            response=response,
            diagnostics=diagnostics,
            safety=safety,
            length_metrics=length_metrics,
            fallback=fallback
        )