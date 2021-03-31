package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api"
)

const token = "1665345356:AAHdyRmnKIANP0wbgzCuDWXn59RzZ6bqUcU" //put your token here
var homepath string = ""
var platform string = ""

func init() {
	if os.Getenv("HOMEPATH") == "" {
		homepath = os.Getenv("HOME")
		platform = "unix"
	} else {
		homepath = os.Getenv("HOMEPATH")
		platform = "win"
	}
}
func main() {
	basePath := filepath.Join(homepath, "Desktop/circuit-symbol-detector")
	scriptPath := filepath.Join(basePath, "img.py")
	imgPath := filepath.Join(basePath, "img.jpeg")
	componentFolder := filepath.Join(basePath, "dump")
	pythonCall := "python3 " + scriptPath + " -p " + imgPath

	fmt.Println("server running...")
	fmt.Println("basepath:", basePath)
	fmt.Println("scriptpath:", scriptPath)
	fmt.Println("imgpath:", imgPath)
	fmt.Println("componentfolderpath:", componentFolder)

	bot, err := tgbotapi.NewBotAPI(token)
	if err != nil {
		log.Panic(err)
	}
	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60
	updates, err := bot.GetUpdatesChan(u)

	for update := range updates {
		if update.Message.Text != "" {
			fmt.Println(update.Message.From, "sent:", update.Message.Text)
			continue
		}

		/*IMAGE*/
		p := (update.Message.Photo)
		if p != nil {
			fmt.Println("image recieved from:", update.Message.From)
			for _, pic := range *p {
				id := tgbotapi.FileConfig{(pic.FileID)}
				f, _ := bot.GetFile(id)
				DownloadFile("img.jpeg", f.Link(token))
				fmt.Println("downloaded image..")

			}
			time.Sleep(time.Second)
			var cmd *exec.Cmd
			if platform == "win" {
				fmt.Println("calling powershell python")
				cmd = exec.Command("powershell.exe", pythonCall)
			} else {
				fmt.Println("calling bash python")
				fmt.Println("python3 " + scriptPath + " -p " + imgPath)
				cmd = exec.Command("python3", scriptPath+" -p "+imgPath)
			}

			err := cmd.Run()
			if err != nil {
				fmt.Println(err)
			}
			fmt.Println(cmd.Output())
			f, err := ioutil.ReadDir(componentFolder)
			if err != nil {
				msg := tgbotapi.NewMessage(update.Message.Chat.ID, "that didnt work")
				bot.Send(msg)
			}
			fmt.Println("detected components..")
			for _, comp := range f {
				fmt.Println(comp.Name())
				path := filepath.Join(componentFolder, comp.Name())
				pic := tgbotapi.NewPhotoUpload(update.Message.Chat.ID, path)
				pic.Caption = strings.Split(comp.Name(), ".")[0]
				bot.Send(pic)
				os.Remove(path)
			}
			//os.Remove(imgPath)
		}

	}
}

//DownloadFile from `url` to `filepath`
func DownloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}
