package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Usage struct {
	PromptTokens     int `json:"promptTokens"`
	CompletionTokens int `json:"completionTokens"`
	TotalTokens      int `json:"totalTokens"`
}

func (u *Usage) Add(other Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
}

type LLMResponse struct {
	Message      Message
	FinishReason string
	Usage        Usage
}

type LLMClient interface {
	Chat(ctx context.Context, messages []Message) (LLMResponse, error)
	Model() string
}

type DeepSeekClient struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
}

type deepSeekRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	Thinking    *thinking `json:"thinking,omitempty"`
}

type thinking struct {
	Type string `json:"type"`
}

type deepSeekResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage deepSeekUsage `json:"usage"`
}

type deepSeekUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func NewDeepSeekClient(baseURL, apiKey, model string) *DeepSeekClient {
	return &DeepSeekClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		client:  &http.Client{Timeout: 180 * time.Second},
	}
}

func (d *DeepSeekClient) Model() string { return d.model }

func (d *DeepSeekClient) Chat(ctx context.Context, messages []Message) (LLMResponse, error) {
	temp := 0.0
	body, err := json.Marshal(deepSeekRequest{
		Model:       d.model,
		Messages:    messages,
		Temperature: &temp,
		Thinking:    &thinking{Type: "disabled"},
	})
	if err != nil {
		return LLMResponse{}, fmt.Errorf("сборка запроса DeepSeek: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return LLMResponse{}, fmt.Errorf("создание запроса DeepSeek: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+d.apiKey)

	resp, err := d.client.Do(req)
	if err != nil {
		return LLMResponse{}, fmt.Errorf("DeepSeek: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return LLMResponse{}, fmt.Errorf("чтение ответа DeepSeek: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		var apiErr struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(data, &apiErr) == nil && apiErr.Error.Message != "" {
			return LLMResponse{}, fmt.Errorf("DeepSeek: %s", apiErr.Error.Message)
		}
		return LLMResponse{}, fmt.Errorf("DeepSeek вернул статус %d", resp.StatusCode)
	}

	var out deepSeekResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return LLMResponse{}, fmt.Errorf("разбор ответа DeepSeek: %w", err)
	}
	if len(out.Choices) == 0 {
		return LLMResponse{}, errors.New("DeepSeek вернул пустой choices")
	}

	return LLMResponse{
		Message:      out.Choices[0].Message,
		FinishReason: out.Choices[0].FinishReason,
		Usage: Usage{
			PromptTokens:     out.Usage.PromptTokens,
			CompletionTokens: out.Usage.CompletionTokens,
			TotalTokens:      out.Usage.TotalTokens,
		},
	}, nil
}
