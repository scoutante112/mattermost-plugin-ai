// Copyright (c) 2023-present Mattermost, Inc. All Rights Reserved.
// See LICENSE.txt for license information.

package gemini

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"

	"github.com/mattermost/mattermost-plugin-ai/server/llm"
)

// Config holds the configuration for the Gemini provider
type Config struct {
	APIKey        string
	ModelName     string
	MaxTokens     int
	Temperature   float32
	TopP          float32
	TopK          int32
	StopSequences []string
}

// Gemini implements the llm.Provider interface for Google's Gemini API
type Gemini struct {
	config     Config
	httpClient *http.Client
	client     *genai.Client
}

// New creates a new Gemini provider
func New(config Config, httpClient *http.Client) *Gemini {
	return &Gemini{
		config:     config,
		httpClient: httpClient,
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

// GetChatCompletion implements the llm.Provider interface
func (g *Gemini) GetChatCompletion(ctx context.Context, messages []llm.Message, options ...llm.Option) (*llm.Response, error) {
	if g.client == nil {
		if err := g.Initialize(); err != nil {
			return nil, err
		}
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
	if len(g.config.StopSequences) > 0 {
		model.SetStopSequences(g.config.StopSequences)
	}
	
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

// GetEmbedding implements the llm.Provider interface
func (g *Gemini) GetEmbedding(ctx context.Context, input string) ([]float32, error) {
	// Gemini currently doesn't support embeddings through the Go SDK
	// This is a placeholder for when it becomes available
	return nil, errors.New("embedding not supported by Gemini provider")
}
