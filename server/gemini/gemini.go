// Copyright (c) 2023-present Mattermost, Inc. All Rights Reserved.
// See LICENSE.txt for license information.

package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"

	"github.com/mattermost/mattermost-plugin-ai/server/llm"
	"github.com/mattermost/mattermost-plugin-ai/server/metrics"
)

// Config holds the configuration for the Gemini provider
type Config struct {
	APIKey        string   `json:"apiKey"`
	ModelName     string   `json:"modelName"`
	MaxTokens     int      `json:"maxTokens"`
	Temperature   float32  `json:"temperature"`
	TopP          float32  `json:"topP"`
	TopK          int32    `json:"topK"`
	StopSequences []string `json:"stopSequences"`
}

// Gemini implements the llm.LanguageModel interface for Google's Gemini API
type Gemini struct {
	config     Config
	httpClient *http.Client
	client     *genai.Client
	metrics    *metrics.LlmMetrics
}

// New creates a new Gemini provider
func New(serviceConfig llm.ServiceConfig, httpClient *http.Client, metrics *metrics.LlmMetrics) llm.LanguageModel {
	var config Config
	if err := json.Unmarshal(serviceConfig.Parameters, &config); err != nil {
		// Just use default config if unmarshal fails
	}
	
	return &Gemini{
		config:     config,
		httpClient: httpClient,
		metrics:    metrics,
	}
}

func (g *Gemini) Initialize() error {
	if g.config.APIKey == "" {
		return errors.New("Gemini API key is required")
	}

	if g.config.ModelName == "" {
		g.config.ModelName = "gemini-pro"
	}

	client, err := genai.NewClient(context.Background(), option.WithAPIKey(g.config.APIKey), option.WithHTTPClient(g.httpClient))
	if err != nil {
		return fmt.Errorf("failed to create Gemini client: %w", err)
	}

	g.client = client
	return nil
}

// GetName returns the name of the provider
func (g *Gemini) GetName() string {
	return "gemini"
}

// GetModelName returns the model name being used
func (g *Gemini) GetModelName() string {
	return g.config.ModelName
}

// GetChatCompletion implements the llm.LanguageModel interface
func (g *Gemini) GetChatCompletion(ctx context.Context, messages []llm.Message, options ...llm.Option) (*llm.Response, error) {
	if g.client == nil {
		if err := g.Initialize(); err != nil {
			return nil, err
		}
	}

	if g.metrics != nil {
		g.metrics.ObserveRequest()
		defer func() {
			g.metrics.ObserveResponse()
		}()
	}

	opts := llm.NewOptions(options...)
	
	model := g.client.GenerativeModel(g.config.ModelName)
	
	// Configure the model based on our config and options
	model.SetTemperature(float64(g.config.Temperature))
	if g.config.MaxTokens > 0 {
		model.SetMaxOutputTokens(int32(g.config.MaxTokens))
	}
	if g.config.TopP > 0 {
		model.SetTopP(float64(g.config.TopP))
	}
	if g.config.TopK > 0 {
		model.SetTopK(g.config.TopK)
	}
	
	// Note: Gemini Go SDK doesn't have SetStopSequences method
	// We'll skip this functionality for now
	
	// Convert Mattermost messages to Gemini chat messages
	var geminiContents []*genai.Content
	for _, msg := range messages {
		content := &genai.Content{
			Parts: []genai.Part{
				genai.Text(msg.Content),
			},
		}
		
		switch msg.Role {
		case llm.RoleUser:
			content.Role = "user"
		case llm.RoleAssistant:
			content.Role = "model"
		case llm.RoleSystem:
			// Gemini doesn't have a system role, so we'll use user role with a prefix
			content.Role = "user"
			content.Parts = []genai.Part{
				genai.Text("System instruction: " + msg.Content),
			}
		}
		
		geminiContents = append(geminiContents, content)
	}
	
	// Generate content
	resp, err := model.GenerateContent(ctx, geminiContents...)
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}
	
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("no response generated")
	}
	
	// Extract the response text
	responseText := ""
	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			responseText += string(textPart)
		}
	}
	
	return &llm.Response{
		Content: responseText,
	}, nil
}

// GetEmbedding implements the llm.LanguageModel interface
func (g *Gemini) GetEmbedding(ctx context.Context, input string) ([]float32, error) {
	// Gemini currently doesn't support embeddings through the Go SDK
	// This is a placeholder for when it becomes available
	return nil, errors.New("embedding not supported by Gemini provider")
}

// GetTranscription implements the llm.LanguageModel interface
func (g *Gemini) GetTranscription(ctx context.Context, audioData []byte, prompt string) (string, error) {
	// Gemini currently doesn't support audio transcription through the Go SDK
	return "", errors.New("transcription not supported by Gemini provider")
}

// GetVision implements the llm.LanguageModel interface
func (g *Gemini) GetVision(ctx context.Context, messages []llm.Message, options ...llm.Option) (*llm.Response, error) {
	// Check if we have a vision-capable model
	if !strings.Contains(g.config.ModelName, "vision") {
		// Switch to vision model if available
		g.config.ModelName = "gemini-pro-vision"
	}
	
	// Process the messages to include images
	var geminiContents []*genai.Content
	for _, msg := range messages {
		content := &genai.Content{
			Parts: []genai.Part{},
		}
		
		// Add text content
		if msg.Content != "" {
			content.Parts = append(content.Parts, genai.Text(msg.Content))
		}
		
		// Add image attachments if any
		for _, attachment := range msg.Attachments {
			if strings.HasPrefix(attachment.MimeType, "image/") {
				imgData := genai.ImageData{
					MIMEType: attachment.MimeType,
					Data:     attachment.Data,
				}
				content.Parts = append(content.Parts, genai.Blob{Data: imgData})
			}
		}
		
		switch msg.Role {
		case llm.RoleUser:
			content.Role = "user"
		case llm.RoleAssistant:
			content.Role = "model"
		case llm.RoleSystem:
			content.Role = "user"
			// Prepend system instruction to first part if it's text
			if len(content.Parts) > 0 {
				if textPart, ok := content.Parts[0].(genai.Text); ok {
					content.Parts[0] = genai.Text("System instruction: " + string(textPart))
				} else {
					// Insert system instruction at the beginning
					newParts := make([]genai.Part, len(content.Parts)+1)
					newParts[0] = genai.Text("System instruction: ")
					copy(newParts[1:], content.Parts)
					content.Parts = newParts
				}
			} else {
				content.Parts = append(content.Parts, genai.Text("System instruction: "))
			}
		}
		
		geminiContents = append(geminiContents, content)
	}
	
	// Use the same completion logic as GetChatCompletion
	if g.client == nil {
		if err := g.Initialize(); err != nil {
			return nil, err
		}
	}
	
	model := g.client.GenerativeModel(g.config.ModelName)
	
	// Configure the model
	model.SetTemperature(float64(g.config.Temperature))
	if g.config.MaxTokens > 0 {
		model.SetMaxOutputTokens(int32(g.config.MaxTokens))
	}
	if g.config.TopP > 0 {
		model.SetTopP(float64(g.config.TopP))
	}
	if g.config.TopK > 0 {
		model.SetTopK(g.config.TopK)
	}
	
	// Generate content
	resp, err := model.GenerateContent(ctx, geminiContents...)
	if err != nil {
		return nil, fmt.Errorf("failed to generate vision content: %w", err)
	}
	
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("no vision response generated")
	}
	
	// Extract the response text
	responseText := ""
	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			responseText += string(textPart)
		}
	}
	
	return &llm.Response{
		Content: responseText,
	}, nil
}
