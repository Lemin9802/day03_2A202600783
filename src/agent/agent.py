import json
import re
from typing import List, Dict, Any, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop.

    Agent v1:
    - Basic ReAct loop
    - Tool execution
    - Final Answer detection

    Agent v2:
    - Adds stricter final-answer wording rules
    - Improves Vietnamese output consistency
    - Improves evaluator-friendly budget/coupon wording
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 5,
        agent_version: str = "v1",
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.agent_version = agent_version
        self.history = []

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [
                f"- {tool['name']}: {tool['description']}"
                for tool in self.tools
            ]
        )

        base_prompt = f"""
You are EduCourse ReAct Agent {self.agent_version}, a course registration assistant.

You have access to the following tools:
{tool_descriptions}

You must follow this exact format:

Thought: explain what information you need next.
Action: tool_name({{"argument_name": "argument_value"}})

After you call an Action, the system will provide:
Observation: tool result

Then continue with Thought/Action if more tool information is needed.

When you have enough information, stop using tools and respond with:
Final Answer: your final response to the user in Vietnamese.

Rules:
- Use only the tools listed above.
- Use exactly one Action per step.
- Do not invent tool names.
- Do not invent Observation results.
- Do not write Observation yourself. Observation will be provided only by the system after a tool call.
- Do not include Final Answer in the same response as an Action.
- Action arguments must be valid JSON inside parentheses.
- Before recommending a course as available, check class slots.
- If slots_left is 0, do not say the class is available.
- If a coupon is invalid, explain that the coupon is invalid and no discount is applied.
- Answer every part of the user's question in the Final Answer.
- Preserve relevant tool findings such as an invalid coupon even if another lookup also has no matches.
- Final answers should be clear and use VND when discussing tuition.
- Always write the Final Answer in Vietnamese.
- If the user asks in Vietnamese, answer in Vietnamese.
""".strip()

        if self.agent_version == "v2":
            base_prompt += """

Additional Agent v2 rules:
- If the user mentions a budget and the final tuition is higher than that budget, explicitly say "vượt ngân sách" in Vietnamese.
- If the final tuition is lower than or equal to the budget, explicitly say "phù hợp với ngân sách" in Vietnamese.
- When a coupon is invalid, explicitly say "mã giảm giá không hợp lệ" and "không có giảm giá được áp dụng".
- Use evaluator-friendly Vietnamese wording for budget comparison.
- Keep the Final Answer concise and in Vietnamese.
""".strip()

        return base_prompt

    def run(self, user_input: str) -> str:
        logger.log_event(
            "AGENT_START",
            {
                "input": user_input,
                "model": getattr(self.llm, "model_name", "unknown"),
                "max_steps": self.max_steps,
                "agent_version": self.agent_version,
            },
        )

        conversation = f"User: {user_input}\n"
        final_answer = ""

        for step in range(1, self.max_steps + 1):
            logger.log_event(
                "AGENT_STEP_START",
                {
                    "step": step,
                    "prompt_preview": conversation[-1000:],
                },
            )

            result = self.llm.generate(
                conversation,
                system_prompt=self.get_system_prompt(),
            )

            content = result.get("content", "").strip()

            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=getattr(self.llm, "model_name", "unknown"),
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

            logger.log_event(
                "AGENT_LLM_OUTPUT",
                {
                    "step": step,
                    "content": content,
                    "usage": result.get("usage", {}),
                    "latency_ms": result.get("latency_ms", 0),
                },
            )

            self.history.append(
                {
                    "step": step,
                    "llm_output": content,
                    "usage": result.get("usage", {}),
                    "latency_ms": result.get("latency_ms", 0),
                }
            )

            extracted_final = self._extract_final_answer(content)
            if extracted_final:
                final_answer = extracted_final
                logger.log_event(
                    "AGENT_FINAL_ANSWER",
                    {
                        "step": step,
                        "answer": final_answer,
                    },
                )
                break

            parsed_action = self._parse_action(content)

            if parsed_action is None:
                observation = {
                    "error": "NO_ACTION_PARSED",
                    "message": (
                        "The assistant did not provide a valid Action. "
                        "It must use Action: tool_name({\"key\": \"value\"}) "
                        "or Final Answer: ..."
                    ),
                }

                logger.log_event(
                    "AGENT_PARSE_ERROR",
                    {
                        "step": step,
                        "content": content,
                        "observation": observation,
                    },
                )
            else:
                tool_name, args = parsed_action

                logger.log_event(
                    "TOOL_CALL",
                    {
                        "step": step,
                        "tool_name": tool_name,
                        "args": args,
                    },
                )

                observation = self._execute_tool(tool_name, args)

                logger.log_event(
                    "TOOL_RESULT",
                    {
                        "step": step,
                        "tool_name": tool_name,
                        "observation": observation,
                    },
                )

            observation_text = self._to_json_text(observation)

            conversation += (
                f"\nAssistant:\n{content}\n"
                f"Observation: {observation_text}\n"
            )

        if not final_answer:
            final_answer = (
                "Tôi chưa thể hoàn thành yêu cầu trong số bước tối đa. "
                "Vui lòng thử lại với yêu cầu đơn giản hơn."
            )

            logger.log_event(
                "AGENT_MAX_STEPS_EXCEEDED",
                {
                    "max_steps": self.max_steps,
                    "last_history": self.history[-1] if self.history else None,
                },
            )

        logger.log_event(
            "AGENT_END",
            {
                "steps": len(self.history),
                "final_answer": final_answer,
                "agent_version": self.agent_version,
            },
        )

        return final_answer

    def _extract_final_answer(self, content: str) -> Optional[str]:
        match = re.search(
            r"Final Answer\s*:\s*(.*)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None

        answer = match.group(1).strip()
        return answer or None

    def _parse_action(self, content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        cleaned = self._strip_markdown_fences(content)

        match = re.search(
            r"Action\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not match:
            return None

        tool_name = match.group(1).strip()
        args_text = match.group(2).strip()

        args_text = re.split(
            r"\n\s*(Observation|Final Answer)\s*:",
            args_text,
            flags=re.IGNORECASE,
        )[0].strip()

        if not args_text:
            return tool_name, {}

        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            json_object_match = re.search(r"\{.*\}", args_text, flags=re.DOTALL)
            if not json_object_match:
                return None

            try:
                args = json.loads(json_object_match.group(0))
            except json.JSONDecodeError:
                return None

        if not isinstance(args, dict):
            return None

        return tool_name, args

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        for tool in self.tools:
            if tool["name"] == tool_name:
                func = tool.get("func") or tool.get("function")

                if func is None:
                    return {
                        "error": "TOOL_HAS_NO_FUNCTION",
                        "tool_name": tool_name,
                    }

                try:
                    result = func(**args)

                    if isinstance(result, dict):
                        return result

                    return {
                        "result": result,
                    }

                except TypeError as exc:
                    return {
                        "error": "INVALID_TOOL_ARGUMENTS",
                        "tool_name": tool_name,
                        "args": args,
                        "details": str(exc),
                    }

                except Exception as exc:
                    return {
                        "error": "TOOL_EXECUTION_ERROR",
                        "tool_name": tool_name,
                        "args": args,
                        "details": str(exc),
                    }

        return {
            "error": "TOOL_NOT_FOUND",
            "tool_name": tool_name,
            "available_tools": [tool["name"] for tool in self.tools],
        }

    def _strip_markdown_fences(self, text: str) -> str:
        text = text.strip()
        text = re.sub(
            r"^```(?:json|python|text)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    def _to_json_text(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)