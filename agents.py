import json
import warnings
from typing import Dict, Any
import autogen
from autogen import AssistantAgent

def _last_content(chat_result) -> str:
    if hasattr(chat_result, "chat_history") and chat_result.chat_history:
        last = chat_result.chat_history[-1]
        if hasattr(last, "content"):
            return last.content or ""
        if isinstance(last, dict) and "content" in last:
            return last["content"] or ""
    return ""

def run_three_agent_debate(metrics: Dict[str, Any], api_key: str) -> Dict[str, str]:
    llm_config = {
        "config_list": [
            {
                "model": "gemini-2.5-flash",
                "api_key": api_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
            }
        ],
        "temperature": 0.35,
        "timeout": 120,
    }

    try:
        risk_analyst = AssistantAgent(
            name="risk_analyst",
            system_message=(
                "You are a conservative financial analyst. Write a compact verdict (80–120 words) covering "
                "concentration risks, sector imbalances, and potential drawdowns. Offer 1–2 cautionary steps."
            ),
            llm_config=llm_config,
        )

        growth_strategist = AssistantAgent(
            name="growth_strategist",
            system_message=(
                "You are a growth-focused strategist. Write a compact verdict (80–120 words) highlighting opportunities, "
                "underweighted growth areas, and 2 practical steps to improve upside."
            ),
            llm_config=llm_config,
        )

        lead_analyst = AssistantAgent(
            name="lead_analyst",
            system_message=(
                "You are the lead analyst. Given two expert notes (risk and growth) and portfolio metrics, write a "
                "final advisory note (120–180 words) balancing risk and opportunity, summarizing metrics, and providing 2-3 steps."
            ),
            llm_config=llm_config,
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=None, # Cleaned config allocation
        )

        metrics_json = json.dumps(metrics, indent=2)

        risk_result = user_proxy.initiate_chat(
            risk_analyst,
            message=f"Portfolio metrics:\n```json\n{metrics_json}\n```\n\nWrite your risk verdict now.",
            max_turns=1,
        )
        risk_verdict = _last_content(risk_result).strip()

        growth_result = user_proxy.initiate_chat(
            growth_strategist,
            message=f"Portfolio metrics:\n```json\n{metrics_json}\n```\n\nWrite your growth verdict now.",
            max_turns=1,
        )
        growth_verdict = _last_content(growth_result).strip()

        lead_msg = (
            f"METRICS:\n```json\n{metrics_json}\n```\n\n"
            f"RISK ANALYST NOTE:\n{risk_verdict}\n\n"
            f"GROWTH STRATEGIST NOTE:\n{growth_verdict}\n\n"
            "Write the final advisory note."
        )
        lead_result = user_proxy.initiate_chat(lead_analyst, message=lead_msg, max_turns=1)
        lead_verdict = _last_content(lead_result).strip()

        return {
            "risk_verdict": risk_verdict,
            "growth_verdict": growth_verdict,
            "lead_verdict": lead_verdict,
            "final_note": lead_verdict,
        }

    except Exception as e:
        warnings.warn(f"Gemini multi-agent simulation failure: {e}")
        raise e
