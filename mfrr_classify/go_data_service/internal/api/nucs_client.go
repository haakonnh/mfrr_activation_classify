package api

import (
	// dotenv
	"log"
	"os"

	"github.com/joho/godotenv"
)

func Init() {
	// Load environment variables from .env file
	err := godotenv.Load()

	if err != nil {
		log.Println("No .env file found or error loading .env file")
	}

	apiKey := os.Getenv("NUCS_API_KEY")
	nucsBaseURL := os.Getenv("NUCS_BASE_URL")
	log.Println("NUCS_BASE_URL:", nucsBaseURL)
	log.Println("NUCS_API_KEY:", apiKey)
}
