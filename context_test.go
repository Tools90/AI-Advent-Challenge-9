package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

type fakeLLM struct {
	calls    int
	requests [][]Message
}

func (f *fakeLLM) Chat(ctx context.Context, messages []Message) (LLMResponse, error) {
	f.calls++
	f.requests = append(f.requests, append([]Message(nil), messages...))
	content := fmt.Sprintf("reply-%d", f.calls)
	if len(messages) > 0 && messages[0].Role == "system" && strings.Contains(messages[0].Content, "сжимаешь историю") {
		content = "SUMMARY"
	}
	if len(messages) > 0 && messages[0].Role == "system" && strings.Contains(messages[0].Content, "судья") {
		content = `{"correct":true,"reason":"ok"}`
	}
	return LLMResponse{
		Message: Message{Role: "assistant", Content: content},
		Usage:   Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}, nil
}

func (f *fakeLLM) Model() string { return "fake" }

func fillPairs(t *testing.T, session *Session, llm LLMClient, count int, compression bool) {
	t.Helper()
	for i := 0; i < count; i++ {
		if _, err := session.Chat(context.Background(), llm, fmt.Sprintf("u%d", i), compression); err != nil {
			t.Fatalf("pair %d: %v", i, err)
		}
	}
}

func TestSummaryTriggerAndRequest(t *testing.T) {
	session := NewSession()
	llm := &fakeLLM{}
	fillPairs(t, session, llm, 10, true)

	state := session.State()
	if state.Pairs != 10 || state.SummarizedPairs != 5 || state.SummaryUpdates != 1 {
		t.Fatalf("state = %+v", state)
	}
	if state.Summary != "SUMMARY" {
		t.Errorf("summary = %q", state.Summary)
	}
	if len(state.Messages) != 20 {
		t.Errorf("messages = %d", len(state.Messages))
	}

	llm.requests = nil
	if _, err := session.Chat(context.Background(), llm, "next", true); err != nil {
		t.Fatal(err)
	}
	request := llm.requests[0]
	if len(request) != 12 {
		t.Fatalf("compressed request messages = %d, want 12", len(request))
	}
	if !strings.Contains(request[0].Content, "SUMMARY") {
		t.Errorf("system does not contain summary: %q", request[0].Content)
	}
	if request[1].Content != "u5" {
		t.Errorf("first history message = %q, want u5", request[1].Content)
	}
	if request[len(request)-1].Content != "next" {
		t.Errorf("last message = %q", request[len(request)-1].Content)
	}
}

func TestNoCompressionUsesFullHistory(t *testing.T) {
	session := NewSession()
	llm := &fakeLLM{}
	fillPairs(t, session, llm, 10, false)

	state := session.State()
	if state.Summary != "" || state.SummarizedPairs != 0 || state.SummaryUpdates != 0 {
		t.Fatalf("summary should be empty: %+v", state)
	}

	llm.requests = nil
	if _, err := session.Chat(context.Background(), llm, "next", false); err != nil {
		t.Fatal(err)
	}
	request := llm.requests[0]
	if len(request) != 22 {
		t.Fatalf("full request messages = %d, want 22", len(request))
	}
	if request[1].Content != "u0" {
		t.Errorf("first history message = %q", request[1].Content)
	}
}

func TestSecondSummaryAtFifteenPairs(t *testing.T) {
	session := NewSession()
	llm := &fakeLLM{}
	fillPairs(t, session, llm, 15, true)

	state := session.State()
	if state.Pairs != 15 || state.SummarizedPairs != 10 || state.SummaryUpdates != 2 {
		t.Fatalf("state = %+v", state)
	}
}

func TestPendingPairsIncludedBeforeNextSummary(t *testing.T) {
	session := NewSession()
	llm := &fakeLLM{}
	fillPairs(t, session, llm, 13, true)

	state := session.State()
	if state.SummarizedPairs != 5 || state.PendingPairs != 3 {
		t.Fatalf("state = %+v", state)
	}

	llm.requests = nil
	if _, err := session.Chat(context.Background(), llm, "next", true); err != nil {
		t.Fatal(err)
	}
	request := llm.requests[0]
	if len(request) != 18 {
		t.Fatalf("request messages = %d, want 18", len(request))
	}
	if request[1].Content != "u5" {
		t.Errorf("pending start = %q, want u5", request[1].Content)
	}
}

func TestReset(t *testing.T) {
	session := NewSession()
	llm := &fakeLLM{}
	fillPairs(t, session, llm, 10, true)
	session.Reset()
	state := session.State()
	if state.Pairs != 0 || state.Summary != "" || state.SummarizedPairs != 0 || len(state.Messages) != 0 {
		t.Errorf("state after reset = %+v", state)
	}
}

func TestJudge(t *testing.T) {
	session := NewSession()
	llm := &fakeLLM{}
	result, err := session.Judge(context.Background(), llm, "Как зовут?", "Виктор", "Виктор")
	if err != nil {
		t.Fatal(err)
	}
	if !result.Correct || result.Reason != "ok" || result.Usage.TotalTokens == 0 {
		t.Errorf("judge result = %+v", result)
	}
	state := session.State()
	if state.JudgeUsage.TotalTokens != result.Usage.TotalTokens {
		t.Errorf("judge usage = %+v", state.JudgeUsage)
	}
}
