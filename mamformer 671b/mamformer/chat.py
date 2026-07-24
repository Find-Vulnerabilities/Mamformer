"""
Mamformer Chat / Conversation System
======================================
Multi-turn dialogue with conversation history management.

Supports:
  - Multi-turn conversation with automatic history truncation
  - System prompt support
  - Token counting and context window management
  - Streaming generation
  - Chat template formatting (Llama-style and custom)
  - Save/load conversation history

Usage:
    from mamformer.chat import ChatSession

    chat = ChatSession(model, tokenizer, system_prompt="You are helpful.")
    chat.add_user_message("What is Mamba?")
    response = chat.generate_response()
    print(response)

    # Multi-turn
    chat.add_user_message("How is it different from Transformer?")
    response = chat.generate_response()

    # View history
    print(chat.get_history())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Generator

import torch


# ═══════════════════════════════════════════════════════════════════════
# Message Types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    token_count: int = 0  # Cached token count for efficient trimming


# ═══════════════════════════════════════════════════════════════════════
# Chat Templates
# ═══════════════════════════════════════════════════════════════════════

class ChatTemplate:
    """
    Formats conversation messages into model input tokens.

    Default: Llama-style format with special tokens.
    Can be customized for different conversation styles.
    """

    def __init__(
        self,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        system_start: str = "<s>[INST] <<SYS>>\n",
        system_end: str = "\n<</SYS>>\n\n",
        user_start: str = "<s>[INST] ",
        user_end: str = " [/INST]",
        assistant_start: str = " ",
        assistant_end: str = " </s>",
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.system_start = system_start
        self.system_end = system_end
        self.user_start = user_start
        self.user_end = user_end
        self.assistant_start = assistant_start
        self.assistant_end = assistant_end

    def format_message(self, message: Message) -> str:
        """Format a single message according to its role."""
        if message.role == "system":
            return self.system_start + message.content + self.system_end
        elif message.role == "user":
            return self.user_start + message.content + self.user_end
        elif message.role == "assistant":
            return self.assistant_start + message.content + self.assistant_end
        else:
            return message.content

    def format_conversation(self, messages: List[Message]) -> str:
        """Format an entire conversation history into a single prompt string."""
        parts = []
        for msg in messages:
            parts.append(self.format_message(msg))
        return "".join(parts)


# Built-in templates
LLAMA2_CHAT_TEMPLATE = ChatTemplate()

LLAMA3_CHAT_TEMPLATE = ChatTemplate(
    bos_token="<|begin_of_text|>",
    eos_token="<|eos_token|>",
    system_start="<|start_header_id|>system<|end_header_id|>\n\n",
    system_end="<|eot_id|>",
    user_start="<|start_header_id|>user<|end_header_id|>\n\n",
    user_end="<|eot_id|>",
    assistant_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_end="<|eot_id|>",
)

SIMPLE_CHAT_TEMPLATE = ChatTemplate(
    bos_token="",
    eos_token="",
    system_start="System: ",
    system_end="\n",
    user_start="User: ",
    user_end="\n",
    assistant_start="Assistant: ",
    assistant_end="\n",
)

# ── Reflection Templates ──────────────────────────────────────────
# These wrap the base template with reflection instructions

REFLECTION_SYSTEM_PROMPT = (
    "Before answering, think carefully about the question. "
    "Consider multiple perspectives, check for errors in your reasoning, "
    "and only then provide your final answer."
)

THINK_TEMPLATE = ChatTemplate(
    bos_token="<s>",
    eos_token="</s>",
    system_start="<s>[INST] <<SYS>>\n",
    system_end="\n<</SYS>>\n\n",
    user_start="<s>[INST] ",
    user_end=" [/INST]",
    assistant_start="<thinking>\n",
    assistant_end="\n</thinking>\n\n<answer>\n",
)

SELF_CRITIQUE_TEMPLATE = ChatTemplate(
    bos_token="<s>",
    eos_token="</s>",
    system_start="<s>[INST] <<SYS>>\n",
    system_end="\n<</SYS>>\n\n",
    user_start="<s>[INST] ",
    user_end=" [/INST]",
    assistant_start="<draft>\n",
    assistant_end="\n</draft>\n\n<critique>\nReview the above draft for errors, bias, or missing information. Then provide the improved answer:\n</critique>\n\n<final>\n",
)


# ═══════════════════════════════════════════════════════════════════════
# Chat Session
# ═══════════════════════════════════════════════════════════════════════

class ChatSession:
    """
    Manages a multi-turn conversation with history.

    Automatically handles:
      - Token counting and context window limits
      - Old message trimming when context is full
      - System prompt persistence
      - Conversation export/import

    Args:
        model: MamformerForCausalLM instance
        tokenizer: MamformerTokenizer instance
        system_prompt: Optional system prompt
        max_context: Maximum context window in tokens
        template: ChatTemplate for message formatting
        device: Device for tensor operations
    """

    def __init__(
        self,
        model,
        tokenizer,
        system_prompt: Optional[str] = None,
        max_context: int = 4096,
        template: Optional[ChatTemplate] = None,
        device: str = "cpu",
        reflection_mode: str = "none",  # "none" | "think" | "critique"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = max_context
        self.template = template or ChatTemplate()
        self.device = device
        self.reflection_mode = reflection_mode

        # Initialize message history
        self.messages: List[Message] = []
        if system_prompt:
            self.set_system_prompt(system_prompt)

        # Generation settings
        self.temperature: float = 0.7
        self.top_k: int = 50
        self.top_p: float = 0.9
        self.max_new_tokens: int = 512
        self.repetition_penalty: float = 1.0

        # Reflection settings
        self.think_tokens: int = 256       # Max tokens for thinking
        self.critique_tokens: int = 128     # Max tokens for self-critique
        self.show_thinking: bool = False    # Whether to include thinking in history
        self._last_thinking: str = ""       # Stored for inspection

    # ── System Prompt ──────────────────────────────────────────────

    def set_system_prompt(self, prompt: str):
        """Set or replace the system prompt (always at position 0)."""
        token_count = self._count_tokens(prompt)
        msg = Message(role="system", content=prompt, token_count=token_count)
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = msg
        else:
            self.messages.insert(0, msg)

    def get_system_prompt(self) -> Optional[str]:
        """Get the current system prompt, if any."""
        if self.messages and self.messages[0].role == "system":
            return self.messages[0].content
        return None

    # ── Messages ───────────────────────────────────────────────────

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        token_count = self._count_tokens(content)
        self.messages.append(Message(role="user", content=content, token_count=token_count))

    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation."""
        token_count = self._count_tokens(content)
        self.messages.append(Message(role="assistant", content=content, token_count=token_count))

    # ── History Management ──────────────────────────────────────────

    def get_history(self) -> List[dict]:
        """Get conversation history as a list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_formatted_prompt(self) -> str:
        """Get the full formatted prompt text for the current conversation."""
        return self.template.format_conversation(self.messages)

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history, optionally keeping the system prompt."""
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def get_token_count(self) -> int:
        """Get total token count for the current conversation."""
        return sum(msg.token_count for msg in self.messages)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        return len(ids)

    # ── Context Window Trimming ────────────────────────────────────

    def _trim_to_context(self):
        """
        Remove oldest non-system messages until total tokens fit in max_context.

        System prompt is never removed. Messages are removed oldest-first
        (FIFO), ensuring recent conversation context is preserved.
        """
        while self.get_token_count() > self.max_context - self.max_new_tokens:
            # Find oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    self.messages.pop(i)
                    break
            else:
                # Only system prompt remains and still too long
                break

    # ── Generation ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate_response(self) -> str:
        """
        Generate an assistant response for the current conversation.

        Automatically:
          1. Trims history to fit context window
          2. Formats conversation into prompt
          3. Generates response
          4. Adds the response to history
          5. Returns the response text

        Returns:
            Generated assistant response
        """
        # Trim history to fit context
        self._trim_to_context()

        # Build prompt with assistant start (so model knows to generate response)
        prompt = self.template.format_conversation(self.messages)
        if self.messages and self.messages[-1].role == "assistant":
            # If conversation already ends with assistant, trim the end marker
            # to allow continuation rather than starting a duplicate assistant turn
            if self.template.assistant_end and prompt.endswith(self.template.assistant_end):
                prompt = prompt[:-len(self.template.assistant_end)]
        prompt += self.template.assistant_start

        # Encode
        input_ids = self.tokenizer.encode(prompt, add_bos=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Check if prompt alone exceeds context
        if input_tensor.shape[1] > self.max_context:
            raise RuntimeError(
                f"Prompt length ({input_tensor.shape[1]}) exceeds max_context "
                f"({self.max_context}). Reduce conversation length or increase max_context."
            )

        # Generate
        output_ids = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode the generated portion (after the prompt)
        full_output = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        # Extract only the new content (after the prompt)
        prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        if full_output.startswith(prompt_text):
            response = full_output[len(prompt_text):].strip()
        else:
            response = full_output.strip()

        # Clean up: stop at the first user/system marker if model continues
        response = self._clean_response(response)

        # Add to history
        self.add_assistant_message(response)

        return response

    @torch.no_grad()
    def generate_stream(self) -> Generator[str, None, None]:
        """
        Generate a response token by token (streaming).

        Yields each decoded text chunk as it is generated.
        Useful for real-time display.

        Yields:
            Text chunks as they are decoded
        """
        self._trim_to_context()

        prompt = self.template.format_conversation(self.messages)
        if self.messages and self.messages[-1].role == "assistant":
            if self.template.assistant_end and prompt.endswith(self.template.assistant_end):
                prompt = prompt[:-len(self.template.assistant_end)]
        prompt += self.template.assistant_start

        input_ids = self.tokenizer.encode(prompt, add_bos=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        if input_tensor.shape[1] > self.max_context:
            raise RuntimeError("Prompt exceeds max_context")

        generated = input_tensor.clone()
        cache = None
        full_response = ""

        for _ in range(self.max_new_tokens):
            outputs = self.model(
                input_ids=generated[:, -1:] if cache is not None else generated,
                use_cache=True,
                cache=cache,
            )

            logits = outputs["logits"][:, -1, :]
            cache = outputs.get("cache")

            # Temperature scaling
            if self.temperature > 0 and self.temperature != 1.0:
                logits = logits / self.temperature

            # Top-k
            if self.top_k > 0:
                k = min(self.top_k, logits.size(-1))
                top_k_vals, _ = torch.topk(logits, k, dim=-1)
                logits = logits.masked_fill(logits < top_k_vals[:, -1:], float("-inf"))

            # Top-p
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumsum > self.top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(mask, float("-inf"))

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            # Decode current token
            token_id = next_token[0, 0].item()
            chunk = self.tokenizer.decode([token_id], skip_special_tokens=True)
            full_response += chunk
            yield chunk

            if token_id == self.tokenizer.eos_token_id:
                break

        # Add complete response to history
        response = self._clean_response(full_response)
        if response:
            self.add_assistant_message(response)

    # ── Reflection Generation ────────────────────────────────────

    @torch.no_grad()
    def generate_with_reflection(self) -> str:
        """
        Generate response with language-level self-reflection.

        Modes:
          - "none": Standard generation (same as generate_response)
          - "think": Model thinks silently, then answers.
                     Only the answer is stored in history.
          - "critique": Model drafts → critiques → finalizes.
                        Only the final answer is stored.
        """
        if self.reflection_mode == "none":
            return self.generate_response()

        elif self.reflection_mode == "think":
            return self._generate_with_thinking()

        elif self.reflection_mode == "critique":
            return self._generate_with_critique()

        else:
            return self.generate_response()

    def _generate_with_thinking(self) -> str:
        """
        Think mode: model reasons internally before answering.

        Uses <thinking>...</thinking><answer>...</answer> format.
        EOS is suppressed during generation so the model can transition
        from thinking to answering naturally. max_new_tokens caps total length.
        """
        self._trim_to_context()

        prompt = THINK_TEMPLATE.format_conversation(self.messages)
        prompt_clean = self._clean_prompt_for_template(prompt, THINK_TEMPLATE)

        input_ids = self.tokenizer.encode(prompt_clean, add_bos=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Suppress EOS so model continues through </thinking> into <answer>
        output_ids = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=self.think_tokens + self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=None,  # Suppress early stopping
        )

        full_output = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        response = full_output[len(prompt_text):] if full_output.startswith(prompt_text) else full_output

        # Extract thinking and answer
        thinking, answer = self._parse_thinking_response(response)

        # Store thinking for inspection
        self._last_thinking = thinking

        # Build history entry
        if self.show_thinking:
            history_text = f"[thinking]\n{thinking}\n[/thinking]\n\n{answer}"
        else:
            history_text = answer

        self.add_assistant_message(self._clean_response(history_text))
        return answer

    def _generate_with_critique(self) -> str:
        """
        Critique mode: draft → self-critique → final answer.

        Uses <draft>...</draft><critique>...</critique><final>...</final> format.
        """
        self._trim_to_context()

        prompt = SELF_CRITIQUE_TEMPLATE.format_conversation(self.messages)
        prompt_clean = self._clean_prompt_for_template(prompt, SELF_CRITIQUE_TEMPLATE)

        input_ids = self.tokenizer.encode(prompt_clean, add_bos=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        total_gen_tokens = self.max_new_tokens + self.critique_tokens + self.max_new_tokens
        output_ids = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=total_gen_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        full_output = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        response = full_output[len(prompt_text):] if full_output.startswith(prompt_text) else full_output

        # Parse draft, critique, final
        final_answer = self._parse_critique_response(response)

        if not final_answer:
            final_answer = response  # Fallback to raw output

        self._last_thinking = response  # Store full chain for inspection
        self.add_assistant_message(self._clean_response(final_answer))
        return final_answer

    def _parse_thinking_response(self, text: str) -> Tuple[str, str]:
        """Extract thinking and answer from think-mode output."""
        thinking = ""
        answer = text

        # Try <thinking>...</thinking><answer>...</answer>
        if "<thinking>" in text and "</thinking>" in text:
            think_start = text.find("<thinking>") + len("<thinking>")
            think_end = text.find("</thinking>")
            thinking = text[think_start:think_end].strip()

            # Extract answer after </thinking>
            after_think = text[think_end + len("</thinking>"):]
            if "<answer>" in after_think:
                ans_start = after_think.find("<answer>") + len("<answer>")
                answer = after_think[ans_start:].strip()
            else:
                answer = after_think.strip()

        return thinking, answer

    def _parse_critique_response(self, text: str) -> str:
        """Extract final answer from critique-mode output."""
        # Try <final>...</final>
        if "<final>" in text:
            final_start = text.rfind("<final>") + len("<final>")
            final_text = text[final_start:].strip()
            if "</final>" in final_text:
                final_text = final_text[:final_text.find("</final>")].strip()
            return final_text

        # Fallback: take everything after the last </critique>
        if "</critique>" in text:
            return text[text.rfind("</critique>") + len("</critique>"):].strip()

        return text.strip()

    def _clean_prompt_for_template(self, prompt: str, template: ChatTemplate) -> str:
        """Clean prompt text to match what the model expects from this template."""
        import re
        # Remove trailing marker patterns from the base template
        # that would conflict with the reflection template's markers
        markers_to_strip = ["<s>[INST] ", " [/INST]", "</s>", "<|eot_id|>",
                           "Assistant: ", "<answer>", "<thinking>", "</thinking>",
                           "<final>", "</final>", "<draft>", "</draft>", "<critique>", "</critique>"]
        cleaned = prompt
        for marker in markers_to_strip:
            if cleaned.strip().endswith(marker.strip()):
                cleaned = cleaned[:cleaned.rfind(marker)]
        return cleaned.strip()

    def get_last_thinking(self) -> str:
        """Get the thinking/reflection from the last generation."""
        return self._last_thinking

    def _clean_response(self, text: str) -> str:
        """Clean generated text: remove model hallucinated user/system turns."""
        # Stop at common hallucinated markers
        stop_markers = [
            "\nUser:", "\nSystem:", "\n<s>", "\n[INST]", "\n<|start_header_id|>",
            "\nHuman:", "\nAssistant:", "\n\nUser", "User: ",
        ]
        for marker in stop_markers:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx]
        return text.strip()

    # ── Import / Export ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Export conversation to a serializable dict."""
        return {
            "messages": self.get_history(),
            "system_prompt": self.get_system_prompt(),
            "config": {
                "max_context": self.max_context,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "repetition_penalty": self.repetition_penalty,
                "reflection_mode": self.reflection_mode,
            },
        }

    @classmethod
    def from_dict(cls, data: dict, model, tokenizer, device: str = "cpu") -> "ChatSession":
        """Restore a conversation from a dict."""
        config = data.get("config", {})
        session = cls(
            model=model,
            tokenizer=tokenizer,
            system_prompt=data.get("system_prompt"),
            max_context=config.get("max_context", 4096),
            device=device,
            reflection_mode=config.get("reflection_mode", "none"),
        )
        session.temperature = config.get("temperature", 0.7)
        session.top_k = config.get("top_k", 50)
        session.top_p = config.get("top_p", 0.9)
        session.max_new_tokens = config.get("max_new_tokens", 512)
        session.repetition_penalty = config.get("repetition_penalty", 1.0)

        # Restore messages
        for msg in data.get("messages", []):
            if msg["role"] != "system":  # System already set
                if msg["role"] == "user":
                    session.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    session.add_assistant_message(msg["content"])

        return session

    def save(self, path: str):
        """Save conversation to a JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, model, tokenizer, device: str = "cpu") -> "ChatSession":
        """Load conversation from a JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, model, tokenizer, device)


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def create_chat(
    model,
    tokenizer,
    system_prompt: Optional[str] = None,
    max_context: Optional[int] = None,
    device: str = "cpu",
) -> ChatSession:
    """
    Convenience function to create a ChatSession.

    Automatically uses the model's config for max_context if not specified.

    Args:
        model: MamformerForCausalLM instance
        tokenizer: MamformerTokenizer instance
        system_prompt: System prompt text
        max_context: Context window size (auto-detected from config if None)
        device: Device string

    Returns:
        Configured ChatSession
    """
    if max_context is None and hasattr(model, 'config'):
        max_context = model.config.generation.max_context

    return ChatSession(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_context=max_context or 4096,
        device=device,
    )
