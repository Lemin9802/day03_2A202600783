# Individual Report: Lab 3 - Chatbot vs ReAct Agent

* **Student Name**: Thái Thị Yến Nhi
* **Student ID**: 2A202600783
* **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

My main contribution was implementing and improving the core ReAct Agent for the EduCourse project. The project compares a normal chatbot baseline with a ReAct Agent that can use tools to answer course registration questions.

* **Modules Implementated**:

  * `src/agent/agent.py`
  * `src/core/provider_factory.py`
  * `run_agent.py`
  * `scripts/smoke_test_providers.py`

* **Code Highlights**:

I implemented the main ReAct loop in `ReActAgent.run()`:

```text
Thought -> Action -> Observation -> Final Answer
```

The agent can:

```text
1. Build a system prompt that includes the available tool list.
2. Ask the LLM to generate a Thought and one Action.
3. Parse the Action using regex and JSON argument extraction.
4. Execute the selected tool dynamically.
5. Support both tool registry formats: func and function.
6. Append the tool Observation back into the next prompt.
7. Detect and return the Final Answer.
8. Stop safely when max_steps is reached.
```

I also implemented provider switching through `provider_factory.py`, allowing the same ReAct Agent to run with OpenAI or Gemini:

```bash
python run_agent.py --provider openai
python run_agent.py --provider gemini
```

Later, I added Agent v2 as a prompt-improved version of Agent v1. Agent v2 keeps the same ReAct loop but improves output consistency. It asks the model to answer Vietnamese queries in Vietnamese and to use evaluator-friendly phrases such as:

```text
vượt ngân sách
phù hợp với ngân sách
mã giảm giá không hợp lệ
không có giảm giá được áp dụng
```

* **Documentation**:

My code connects three main layers:

```text
LLM Provider -> ReAct Agent -> EduCourse Tools
```

The LLM provider generates the next reasoning step. The ReAct Agent manages the Thought-Action-Observation loop, parses tool calls, executes tools, and feeds observations back to the LLM. The EduCourse tools return deterministic results such as course availability, coupon validity, and final tuition.

---

## II. Debugging Case Study (10 Points)

* **Problem Description**:

One important debugging case was the budget comparison case:

```text
Em muốn học Data Science beginner cuối tuần, ngân sách 2.000.000 VND. Có phù hợp không?
```

The expected behavior was:

```text
1. Find the Data Science beginner weekend course.
2. Check whether the class still has seats.
3. Calculate or identify the tuition.
4. Compare the tuition with the user's budget.
5. Clearly state whether the course is within budget.
```

The agent found the correct course:

```text
DS101 - Data Science Foundation
Tuition: 2.500.000 VND
Budget: 2.000.000 VND
```

However, the evaluator originally failed the case because the answer used a phrase such as:

```text
above your budget
```

instead of a phrase that the evaluator expected, such as:

```text
vượt ngân sách
```

* **Log Source**:

The Agent v2 test showed that the improved prompt produced the expected Vietnamese wording. In the successful run, the agent found `DS101`, checked the class slots, calculated tuition with `coupon_code="NONE"`, and returned:

```text
Khóa học "Data Science Foundation" (DS101) cấp độ beginner, học cuối tuần có sẵn 3 chỗ trống. Học phí là 2.500.000 VND. Mức học phí này vượt ngân sách 2.000.000 VND của bạn.
```

The relevant log events were:

```text
AGENT_START
AGENT_STEP_START
TOOL_CALL: search_courses
TOOL_RESULT: DS101 found
TOOL_CALL: check_class_slots
TOOL_RESULT: slots_left = 3
TOOL_CALL: calculate_tuition
TOOL_RESULT: final_tuition_vnd = 2500000
AGENT_FINAL_ANSWER
AGENT_END
```

* **Diagnosis**:

The root cause was not a tool execution error. The tool calls were correct and the agent retrieved the correct tuition. The issue was a mismatch between the model's natural wording and the evaluator's keyword-based check.

The evaluator expected specific phrases related to budget comparison. A semantically correct phrase such as `"above your budget"` was not accepted by the evaluator.

* **Solution**:

I added Agent v2 prompt rules:

```text
- If the user mentions a budget and the final tuition is higher than that budget, explicitly say "vượt ngân sách" in Vietnamese.
- If the final tuition is lower than or equal to the budget, explicitly say "phù hợp với ngân sách" in Vietnamese.
- When a coupon is invalid, explicitly say "mã giảm giá không hợp lệ" and "không có giảm giá được áp dụng".
- Keep the Final Answer concise and in Vietnamese.
```

After this update, Agent v2 produced the evaluator-friendly phrase `"vượt ngân sách"` for the budget case.

Additional debugging note:

When testing Agent v2 with Gemini, one run failed because of a Gemini free-tier quota error:

```text
ResourceExhausted: 429
Quota exceeded
```

This was not an agent logic error. The log showed that Gemini successfully completed earlier steps and then failed because the provider quota was exceeded. I continued testing Agent v2 with OpenAI to verify the agent logic.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**:

The `Thought` block helps the agent break a user query into smaller reasoning steps. For example, the Python course query requires the agent to:

```text
1. Search for matching courses.
2. Check class availability.
3. Validate the coupon code.
4. Calculate final tuition.
5. Return a complete answer.
```

A normal chatbot tends to answer directly. It may sound natural, but it cannot reliably verify whether a class exists, whether it still has available seats, or whether a coupon is valid.

2. **Reliability**:

The ReAct Agent was more reliable than the chatbot baseline for multi-step tasks that required real data. It could use observations from tools to ground the final answer.

However, the ReAct Agent can perform worse than a chatbot in some situations:

```text
- It is slower because it may require multiple LLM calls.
- It uses more tokens than a direct chatbot answer.
- It can fail if the LLM formats the Action incorrectly.
- It depends heavily on clear tool descriptions and parser robustness.
- It can be affected by external provider limitations, such as Gemini API quota limits.
```

3. **Observation**:

The Observation step is the most important difference between a chatbot and a ReAct Agent. Observations give the model factual feedback from the environment.

For example:

```json
{
  "course_id": "WEB101",
  "available": false,
  "slots_left": 0
}
```

This observation prevents the agent from incorrectly recommending a full class. Without tool feedback, a normal chatbot may still suggest the course because it does not know the actual catalog state.

---

## IV. Future Improvements (5 Points)

* **Scalability**:

For a production-level system, I would replace the fake Python dictionary course catalog with a real database or API. I would also add more tools for prerequisites, enrollment confirmation, schedule conflicts, and payment status.

* **Safety**:

I would add strict JSON schema validation for every tool call. If the system later connects to real enrollment or payment actions, I would also add a supervisor layer to review high-impact tool calls before execution.

* **Performance**:

The current ReAct Agent uses multiple LLM calls, which increases latency and token usage. Future improvements could include:

```text
- Caching repeated course search results.
- Using structured tool calling instead of regex parsing.
- Adding retry logic only when parsing fails.
- Using a smaller model for simple reasoning steps.
- Adding better cost tracking based on real provider pricing.
```

* **Evaluation**:

I would improve the evaluator by adding Vietnamese normalization and semantic matching. This would reduce false negatives when the answer is correct but uses different wording.

I would also track Agent v1 and Agent v2 separately in evaluation results to measure whether prompt improvements actually increase pass rate.

---
