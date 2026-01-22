package main

import (
	"log"

	"mfrr_classify/go_data_service/internal/api"
	"mfrr_classify/go_data_service/internal/ingest"
)

func main() {
	log.Println("Starting mFRR Go Data Service")
	api.Init()
	service := ingest.NewService()
	if err := service.Run(); err != nil {
		log.Fatal(err)
	}
}
