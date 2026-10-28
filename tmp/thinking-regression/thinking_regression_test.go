//go:build unit

package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestARTEXMiniMaxFlashThinkingNormalization(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		body    string
		want    string
		changed bool
	}{
		{"flash-disabled", "MiniMax-M3.1-Flash", `{"thinking":{"type":"disabled"}}`, `{"thinking":{"type":"adaptive"}}`, true},
		{"flash-enabled", "MiniMax-M3.1-Flash", `{"thinking":{"type":"enabled"}}`, `{"thinking":{"type":"adaptive"}}`, true},
		{"flash-lowercase", "minimax-m3.1-flash", `{"thinking":{"type":"disabled"}}`, `{"thinking":{"type":"adaptive"}}`, true},
		{"flash-disabled-and-none", "MiniMax-M3.1-Flash", `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, `{"thinking":{"type":"adaptive"}}`, true},
		{"flash-adaptive-and-none", "MiniMax-M3.1-Flash", `{"thinking":{"type":"adaptive"},"reasoning_effort":"none"}`, `{"thinking":{"type":"adaptive"}}`, true},
		{"flash-none-only", "MiniMax-M3.1-Flash", `{"reasoning_effort":"none"}`, `{}`, true},
		{"flash-anthropic-none", "MiniMax-M3.1-Flash", `{"output_config":{"effort":"none"}}`, `{"output_config":{}}`, true},
		{"flash-responses-none", "MiniMax-M3.1-Flash", `{"reasoning":{"effort":"none"}}`, `{"reasoning":{}}`, true},
		{"flash-adaptive", "MiniMax-M3.1-Flash", `{"thinking":{"type":"adaptive"}}`, `{"thinking":{"type":"adaptive"}}`, false},
		{"flash-omitted", "MiniMax-M3.1-Flash", `{"model":"artex-auto"}`, `{"model":"artex-auto"}`, false},
		{"flash-preserves-other-fields", "MiniMax-M3.1-Flash", `{"thinking":{"type":"disabled"},"reasoning_effort":"high","temperature":0.2}`, `{"thinking":{"type":"adaptive"},"reasoning_effort":"high","temperature":0.2}`, true},
		{"m27-disabled", "MiniMax-M2.7", `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, false},
		{"unrecognized-minimax-disabled", "MiniMax-M3.1-Flash-Unknown", `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, false},
		{"deepseek-disabled", "deepseek-v4.1-flash", `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, `{"thinking":{"type":"disabled"},"reasoning_effort":"none"}`, false},
		{"deepseek-enabled", "deepseek-v4.1-flash", `{"thinking":{"type":"enabled"}}`, `{"thinking":{"type":"enabled"}}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, changed := NormalizeChineseLLMThinking([]byte(tt.body), tt.model)
			require.Equal(t, tt.changed, changed)
			require.JSONEq(t, tt.want, string(got))
			if !tt.changed {
				require.Equal(t, tt.body, string(got), "no-op must preserve the original body")
			}
		})
	}
}

func TestARTEXRawChatNormalizesThinkingAfterAliasMapping(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, stream := range []bool{false, true} {
		for _, tt := range []struct {
			model        string
			platform     string
			wantThinking string
			wantEffort   bool
		}{
			{"MiniMax-M3.1-Flash", PlatformMiniMax, "adaptive", false},
			{"MiniMax-M2.7", PlatformMiniMax, "disabled", true},
			{"deepseek-v4.1-flash", PlatformOpenCodeGo, "disabled", true},
			{"MiniMax-M3.1-Flash", PlatformOpenAI, "adaptive", false},
			{"deepseek-v4.1-flash", PlatformOpenAI, "disabled", true},
		} {
			t.Run(fmt.Sprintf("stream=%t/model=%s/platform=%s", stream, tt.model, tt.platform), func(t *testing.T) {
				body := []byte(fmt.Sprintf(`{"model":"artex-auto","messages":[{"role":"user","content":"Synthetic summary request."}],"thinking":{"type":"disabled"},"reasoning_effort":"none","stream":%t}`, stream))
				rec := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(rec)
				c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
				c.Request.Header.Set("Content-Type", "application/json")
				contentType := "application/json"
				response := fmt.Sprintf(`{"id":"chatcmpl_thinking","object":"chat.completion","model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":"A valid summary."},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`, tt.model)
				if stream {
					contentType = "text/event-stream"
					response = fmt.Sprintf("data: {\"id\":\"chatcmpl_thinking\",\"object\":\"chat.completion.chunk\",\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A valid summary.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\ndata: [DONE]\n\n", tt.model)
				}
				upstream := &httpUpstreamRecorder{resp: &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{contentType}},
					Body:       io.NopCloser(strings.NewReader(response)),
				}}
				svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}
				account := rawChatCompletionsTestAccount()
				account.Platform = tt.platform
				account.Extra = map[string]any{"openai_responses_mode": "force_chat_completions"}
				account.Credentials["api_protocol"] = "chat_completions"
				account.Credentials["model_mapping"] = map[string]any{"artex-auto": tt.model}

				result, err := svc.ForwardAsChatCompletions(context.Background(), c, account, body, "", "")

				require.NoError(t, err)
				require.NotNil(t, result)
				require.NotNil(t, upstream.lastReq)
				require.Equal(t, "/v1/chat/completions", upstream.lastReq.URL.Path)
				require.Equal(t, tt.model, gjson.GetBytes(upstream.lastBody, "model").String())
				require.Equal(t, tt.wantThinking, gjson.GetBytes(upstream.lastBody, "thinking.type").String())
				require.Equal(t, tt.wantEffort, gjson.GetBytes(upstream.lastBody, "reasoning_effort").Exists())
				if tt.wantEffort {
					require.Equal(t, "none", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
				}
				require.Contains(t, rec.Body.String(), "A valid summary.")
			})
		}
	}
}
