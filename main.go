package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web/index.html
var webFS embed.FS

type config struct {
	keyFile string
	apiKey  string
	baseURL string
	model   string
	addr    string
}

type server struct {
	cfg      config
	sessions map[string]*Session
	llm      LLMClient
	chatMu   sync.Mutex
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Printf("config: %v", err)
	}

	cfg := loadConfig()
	if cfg.apiKey == "" {
		log.Printf("deepseek: ключ не задан в %s", cfg.keyFile)
	}

	llm := NewDeepSeekClient(cfg.baseURL, cfg.apiKey, cfg.model)
	s := &server{
		cfg:      cfg,
		sessions: map[string]*Session{"no": NewSession(), "yes": NewSession()},
		llm:      llm,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /api/state", s.state)
	mux.HandleFunc("POST /api/chat", s.chat)
	mux.HandleFunc("POST /api/judge", s.judge)
	mux.HandleFunc("POST /api/reset", s.reset)

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("сервер запущен: http://localhost%s (модель: %s)", cfg.addr, cfg.model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("останавливаюсь...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func (s *server) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "index.html не найден", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":     "ok",
		"model":      s.cfg.model,
		"configured": s.cfg.apiKey != "",
	})
}

func (s *server) state(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("session")
	session, ok := s.sessionByID(id)
	if !ok {
		writeError(w, http.StatusBadRequest, "неизвестная сессия")
		return
	}
	s.chatMu.Lock()
	defer s.chatMu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"model":      s.cfg.model,
		"configured": s.cfg.apiKey != "",
		"session":    session.State(),
	})
}

func (s *server) sessionByID(id string) (*Session, bool) {
	if id == "" {
		id = "no"
	}
	session, ok := s.sessions[id]
	return session, ok
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message string `json:"message"`
		Session string `json:"session"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	message := strings.TrimSpace(body.Message)
	if message == "" {
		writeError(w, http.StatusBadRequest, "сообщение пустое")
		return
	}
	if len([]rune(message)) > 4000 {
		writeError(w, http.StatusBadRequest, "сообщение слишком длинное")
		return
	}

	sessionID := body.Session
	if sessionID == "" {
		sessionID = "no"
	}
	session, ok := s.sessionByID(sessionID)
	if !ok {
		writeError(w, http.StatusBadRequest, "неизвестная сессия")
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	result, err := session.Chat(ctx, s.llm, message, sessionID == "yes")
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *server) judge(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Session  string `json:"session"`
		Question string `json:"question"`
		Expected string `json:"expected"`
		Answer   string `json:"answer"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	session, ok := s.sessionByID(body.Session)
	if !ok {
		writeError(w, http.StatusBadRequest, "неизвестная сессия")
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	result, err := session.Judge(ctx, s.llm, body.Question, body.Expected, body.Answer)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *server) reset(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("session")
	session, ok := s.sessionByID(id)
	if !ok {
		writeError(w, http.StatusBadRequest, "неизвестная сессия")
		return
	}
	s.chatMu.Lock()
	defer s.chatMu.Unlock()
	session.Reset()
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func loadConfig() config {
	keyFile := envOr("DEEPSEEK_API_KEY_FILE", "api_key.txt")
	port := envOr("PORT", "8088")
	return config{
		keyFile: keyFile,
		apiKey:  readSecret(keyFile, "sk-your-key-here"),
		baseURL: strings.TrimRight(envOr("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), "/"),
		model:   envOr("DEEPSEEK_MODEL", "deepseek-flash"),
		addr:    ":" + port,
	}
}

func readSecret(path, placeholder string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	value := unquote(strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff")))
	if value == placeholder || strings.Contains(value, "your-") {
		return ""
	}
	return value
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return fmt.Errorf("некорректный JSON: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("файл %s не найден, использую переменные окружения", path)
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.TrimPrefix(key, "export "))
		value = unquote(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
