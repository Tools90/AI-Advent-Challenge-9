package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	LastPairs     = 5
	SummaryEvery  = 5
	systemPrompt  = "Ты — полезный ассистент. Отвечай по-русски, кратко и по делу."
	summaryPrompt = "Ты сжимаешь историю диалога. Сохрани все конкретные детали: имена, сроки, бюджеты, ограничения, принятые решения, названия технологий и географию. Не добавляй того, чего не было. Верни только краткое summary."
)

type Session struct {
	messages         []Message
	summary          string
	summarizedPairs  int
	compression      bool
	chatUsage        Usage
	summaryUsage     Usage
	judgeUsage       Usage
	summaryUpdates   int
	lastSummaryError string
}

type SessionState struct {
	Messages        []Message `json:"messages"`
	Summary         string    `json:"summary"`
	Pairs           int       `json:"pairs"`
	SummarizedPairs int       `json:"summarizedPairs"`
	PendingPairs    int       `json:"pendingPairs"`
	Compression     bool      `json:"compression"`
	ChatUsage       Usage     `json:"chatUsage"`
	SummaryUsage    Usage     `json:"summaryUsage"`
	JudgeUsage      Usage     `json:"judgeUsage"`
	TotalUsage      Usage     `json:"totalUsage"`
	SummaryUpdates  int       `json:"summaryUpdates"`
	LastSummaryErr  string    `json:"lastSummaryError,omitempty"`
}

type ChatResult struct {
	Reply           string    `json:"reply"`
	Compression     bool      `json:"compression"`
	Request         []Message `json:"request"`
	TurnUsage       Usage     `json:"turnUsage"`
	ChatUsage       Usage     `json:"chatUsage"`
	SummaryUsage    Usage     `json:"summaryUsage"`
	JudgeUsage      Usage     `json:"judgeUsage"`
	TotalUsage      Usage     `json:"totalUsage"`
	Summary         string    `json:"summary"`
	Pairs           int       `json:"pairs"`
	SummarizedPairs int       `json:"summarizedPairs"`
	PendingPairs    int       `json:"pendingPairs"`
	SummaryUpdated  bool      `json:"summaryUpdated"`
	SummaryUpdates  int       `json:"summaryUpdates"`
}

func NewSession() *Session {
	return &Session{}
}

func (s *Session) Reset() {
	s.messages = nil
	s.summary = ""
	s.summarizedPairs = 0
	s.compression = false
	s.chatUsage = Usage{}
	s.summaryUsage = Usage{}
	s.judgeUsage = Usage{}
	s.summaryUpdates = 0
	s.lastSummaryError = ""
}

func (s *Session) totalUsage() Usage {
	var total Usage
	total.Add(s.chatUsage)
	total.Add(s.summaryUsage)
	return total
}

func (s *Session) State() SessionState {
	pairs := len(s.messages) / 2
	evicted := pairs - LastPairs
	if evicted < 0 {
		evicted = 0
	}
	pending := evicted - s.summarizedPairs
	if pending < 0 {
		pending = 0
	}
	return SessionState{
		Messages:        append([]Message(nil), s.messages...),
		Summary:         s.summary,
		Pairs:           pairs,
		SummarizedPairs: s.summarizedPairs,
		PendingPairs:    pending,
		Compression:     s.compression,
		ChatUsage:       s.chatUsage,
		SummaryUsage:    s.summaryUsage,
		JudgeUsage:      s.judgeUsage,
		TotalUsage:      s.totalUsage(),
		SummaryUpdates:  s.summaryUpdates,
		LastSummaryErr:  s.lastSummaryError,
	}
}

func (s *Session) Chat(ctx context.Context, llm LLMClient, user string, compression bool) (ChatResult, error) {
	s.compression = compression
	pairs := len(s.messages) / 2
	lastStart := pairs - LastPairs
	if lastStart < 0 {
		lastStart = 0
	}
	pendingStart := s.summarizedPairs
	if pendingStart > lastStart {
		pendingStart = lastStart
	}

	request := s.buildRequest(user, compression, pendingStart, lastStart, pairs)
	response, err := llm.Chat(ctx, request)
	if err != nil {
		return ChatResult{}, err
	}
	if strings.TrimSpace(response.Message.Content) == "" {
		return ChatResult{}, fmt.Errorf("модель вернула пустой ответ")
	}

	s.messages = append(s.messages,
		Message{Role: "user", Content: user},
		Message{Role: "assistant", Content: response.Message.Content},
	)
	s.chatUsage.Add(response.Usage)

	result := ChatResult{
		Reply:           response.Message.Content,
		Compression:     compression,
		Request:         request,
		TurnUsage:       response.Usage,
		Summary:         s.summary,
		Pairs:           len(s.messages) / 2,
		SummarizedPairs: s.summarizedPairs,
		ChatUsage:       s.chatUsage,
		SummaryUsage:    s.summaryUsage,
		JudgeUsage:      s.judgeUsage,
		TotalUsage:      s.totalUsage(),
		SummaryUpdates:  s.summaryUpdates,
	}

	if compression {
		updated, err := s.maybeSummarize(ctx, llm)
		if err != nil {
			s.lastSummaryError = err.Error()
		} else {
			s.lastSummaryError = ""
		}
		result.SummaryUpdated = updated
		result.Summary = s.summary
		result.SummarizedPairs = s.summarizedPairs
		result.SummaryUpdates = s.summaryUpdates
		result.SummaryUsage = s.summaryUsage
		result.JudgeUsage = s.judgeUsage
		result.TotalUsage = s.totalUsage()
		result.PendingPairs = pendingPairs(len(s.messages)/2, s.summarizedPairs)
	}
	return result, nil
}

func (s *Session) buildRequest(user string, compression bool, pendingStart, lastStart, pairs int) []Message {
	system := systemPrompt
	if compression && strings.TrimSpace(s.summary) != "" {
		system += "\n\nКраткое содержание предыдущей части диалога:\n" + s.summary
	}

	out := []Message{{Role: "system", Content: system}}
	if !compression {
		out = append(out, s.messages...)
		out = append(out, Message{Role: "user", Content: user})
		return out
	}

	for pair := pendingStart; pair < lastStart; pair++ {
		out = append(out, s.messages[pair*2], s.messages[pair*2+1])
	}
	for pair := lastStart; pair < pairs; pair++ {
		out = append(out, s.messages[pair*2], s.messages[pair*2+1])
	}
	out = append(out, Message{Role: "user", Content: user})
	return out
}

type JudgeResult struct {
	Correct      bool   `json:"correct"`
	Reason       string `json:"reason"`
	Usage        Usage  `json:"usage"`
	SessionTotal Usage  `json:"sessionTotal"`
}

func (s *Session) Judge(ctx context.Context, llm LLMClient, question, expected, answer string) (JudgeResult, error) {
	prompt := fmt.Sprintf(
		"Вопрос: %s\nОжидаемый факт: %s\nОтвет ассистента: %s\n\nПроверь, содержит ли ответ ожидаемый факт. Верни только JSON: {\"correct\": true|false, \"reason\": \"кратко\"}.",
		question, expected, answer,
	)
	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: "Ты строгий судья ответов. correct=true, если в ответе присутствует ожидаемый факт, даже если ответ короче или перефразирован. Не требуй дословного совпадения. Примеры: expected «Виктор», answer «Виктор» → true; expected «15 марта», answer «дедлайн 15 марта» → true; expected «только iOS», answer «Android позже, на старте iOS» → true. Верни валидный JSON без markdown."},
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return JudgeResult{}, err
	}
	s.judgeUsage.Add(response.Usage)

	content := strings.TrimSpace(response.Message.Content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return JudgeResult{}, fmt.Errorf("судья вернул не JSON: %s", content)
	}
	var parsed struct {
		Correct bool   `json:"correct"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &parsed); err != nil {
		return JudgeResult{}, fmt.Errorf("разбор ответа судьи: %w", err)
	}
	return JudgeResult{
		Correct:      parsed.Correct,
		Reason:       parsed.Reason,
		Usage:        response.Usage,
		SessionTotal: s.totalUsage(),
	}, nil
}

func (s *Session) maybeSummarize(ctx context.Context, llm LLMClient) (bool, error) {
	totalPairs := len(s.messages) / 2
	evicted := totalPairs - LastPairs
	if evicted < 0 {
		evicted = 0
	}
	if evicted-s.summarizedPairs < SummaryEvery {
		return false, nil
	}

	nextSummary, usage, err := s.summarize(ctx, llm, s.summarizedPairs, evicted)
	if err != nil {
		return false, err
	}
	s.summary = nextSummary
	s.summarizedPairs = evicted
	s.summaryUsage.Add(usage)
	s.summaryUpdates++
	return true, nil
}

func (s *Session) summarize(ctx context.Context, llm LLMClient, startPair, endPair int) (string, Usage, error) {
	var b strings.Builder
	if strings.TrimSpace(s.summary) != "" {
		b.WriteString("Старое summary:\n")
		b.WriteString(s.summary)
		b.WriteString("\n\n")
	}
	b.WriteString("Новые вытесненные пары:\n")
	for pair := startPair; pair < endPair; pair++ {
		b.WriteString("Пользователь: ")
		b.WriteString(s.messages[pair*2].Content)
		b.WriteString("\nАссистент: ")
		b.WriteString(s.messages[pair*2+1].Content)
		b.WriteString("\n\n")
	}

	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: summaryPrompt},
		{Role: "user", Content: b.String()},
	})
	if err != nil {
		return "", Usage{}, err
	}
	summary := strings.TrimSpace(response.Message.Content)
	if summary == "" {
		return "", Usage{}, fmt.Errorf("summary пустой")
	}
	return summary, response.Usage, nil
}

func pendingPairs(pairs, summarized int) int {
	evicted := pairs - LastPairs
	if evicted < 0 {
		evicted = 0
	}
	pending := evicted - summarized
	if pending < 0 {
		return 0
	}
	return pending
}
