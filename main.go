package main

import (
	"database/sql"
	_ "encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	_ "sort"
	_ "strconv"
	_ "strings"
	"time"

	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

type Book struct {
	Attribites map[string]string
}

type Node struct {
	Attribute      string
	Value          string
	Children       []*Node
	IsLeaf         bool
	Classification int
}

func InfoD(labels []int) float64 {
	labelCount := make(map[int]int)
	for _, label := range labels {
		labelCount[label]++
	}
	entropy := 0.0
	for _, count := range labelCount {
		probability := float64(count) / float64(len(labels))
		entropy -= probability * math.Log2(probability)
	}
	return entropy
}

func Split(attribute string, rates map[int]int, books map[int]Book, value string) (map[int]Book, []int) {
	labels := []int{}
	splitedBooks := make(map[int]Book)
	for key, val := range books {
		if val.Attribites[attribute] == value {
			splitedBooks[key] = val
			labels = append(labels, rates[key])
		}
	}
	return splitedBooks, labels
}

func InfoAD(attribute string, rates map[int]int, books map[int]Book, attrVals []string) float64 {
	info := 0.0
	for _, val := range attrVals {
		subset, labels := Split(attribute, rates, books, val)
		prob := float64(len(subset)) / float64(len(rates))
		info += prob * InfoD(labels)
	}
	return info
}

func SplitInfoAD(attribute string, rates map[int]int, books map[int]Book, attrVals []string) float64 {
	splitInfo := 0.0
	for _, val := range attrVals {
		subset, _ := Split(attribute, rates, books, val)
		prob := float64(len(subset)) / float64(len(rates))
		splitInfo -= prob * math.Log2(float64(len(subset))/float64(len(rates)))
	}
	return splitInfo
}

func CalculateGainRatios(rates map[int]int, books map[int]Book, attrValues map[string][]string) map[string]float64 {
	labels := []int{}
	for _, val := range rates {
		labels = append(labels, val)
	}
	ent := InfoD(labels)
	gains := make(map[string]float64)
	splitInfos := make(map[string]float64)
	for key := range attrValues {
		gains[key] = ent - InfoAD(key, rates, books, attrValues[key])
		splitInfos[key] = SplitInfoAD(key, rates, books, attrValues[key])
	}
	gainRatios := make(map[string]float64)
	for key := range gains {
		if splitInfos[key] != 0 {
			gainRatios[key] = gains[key] / splitInfos[key]
		} else {
			gainRatios[key] = 0
		}
	}
	return gainRatios
}

func GetBest(ratios map[string]float64) (string, float64) {
	best := ""
	max := 0.0
	for key, val := range ratios {
		if val > max {
			best = key
			max = val
		}
	}
	return best, max
}

func majorityClass(labels []int) int {
	classCount := make(map[int]int)
	for _, label := range labels {
		classCount[label]++
	}
	maxCount, majority := -1, -1
	for class, count := range classCount {
		if count > maxCount {
			maxCount, majority = count, class
		}
	}
	return majority
}

func SplitDataset(bestAttribute string, val string, rates map[int]int, books map[int]Book) (map[int]int, map[int]Book) {
	for key, value := range books {
		if value.Attribites[bestAttribute] == val {
			delete(value.Attribites, bestAttribute)
		} else {
			delete(books, key)
			delete(rates, key)
		}
	}
	return rates, books
}

func BuildTree(rates map[int]int, books map[int]Book, attrValues map[string][]string) *Node {
	ratios := CalculateGainRatios(rates, books, attrValues)
	labels := []int{}
	for key := range rates {
		labels = append(labels, rates[key])
	}
	bestAttribute, gainRatio := GetBest(ratios)
	if gainRatio == 0 {
		return &Node{IsLeaf: true, Classification: majorityClass(labels)}
	}
	newNode := &Node{Attribute: bestAttribute}
	for _, val := range attrValues[bestAttribute] {
		subRates, subBooks := SplitDataset(bestAttribute, val, rates, books)
		delete(attrValues, bestAttribute)
		child := BuildTree(subRates, subBooks, attrValues)
		child.Value = val
		newNode.Children = append(newNode.Children, child)
	}
	return newNode
}

func Predict(node *Node, book Book) int {
	if node.IsLeaf {
		return node.Classification
	}
	value := book.Attribites[node.Attribute]
	for _, child := range node.Children {
		if child.Value == value {
			return Predict(child, book)
		}
	}
	return 0
}

func Classify(root *Node, rates map[int]int, books map[int]Book) float64 {
	sum := 0
	for key, val := range books {
		prediction := Predict(root, val)
		if prediction == rates[key] {
			sum++
		}
	}
	return float64(sum) / float64(len(books))
}

func main() {
	attributesValues := map[string][]string{"history, historical fiction, biography": {"0", "1"}, "children": {"0", "1"}, "romance": {"0", "1"}, "fantasy, paranormal": {"0", "1"},
		"fiction": {"0", "1"}, "mystery, thriller, crime": {"0", "1"}, "poetry": {"0", "1"}, "young-adult": {"0", "1"}, "non-fiction": {"0", "1"}, "comics, graphic": {"0", "1"},
		"pages": {"Short", "Medium", "Long"}, "age": {"Old", "New", "Mid"}}
	err := godotenv.Load()
	if err != nil {
		fmt.Println("Error loading .env file")
	}
	port := os.Getenv("DB_PORT")
	database := os.Getenv("DB_DB")
	user := os.Getenv("DB_USER")
	pass := os.Getenv("DB_PASSWORD")
	host := os.Getenv("DB_HOST")
	db, err := sql.Open("postgres", fmt.Sprintf("host=%s user=%s password=%s dbname=%s port=%s sslmode=disable", host, user, pass, database, port))
	if err != nil {
		panic(err.Error())
	}
	res, err := db.Query("SELECT * FROM user_rates")
	if err != nil {
		panic(err)
	}
	userrates := make(map[string]map[int]int)
	for res.Next() {
		var bid, rate int
		var uid string
		if err = res.Scan(&uid, &bid, &rate); err != nil {
			panic(err)
		}
		if _, ok := userrates[uid]; ok {
			userrates[uid][bid] = rate
		} else {
			userrates[uid] = make(map[int]int)
			userrates[uid][bid] = rate
		}
	}
	for key, value := range userrates {
		if len(value) < 10 {
			delete(userrates, key)
		}
	}
	res, err = db.Query("SELECT * FROM book_genre")
	if err != nil {
		panic(err)
	}
	bookgenres := make(map[int][]string)
	for res.Next() {
		var bid int
		var genre string
		if err = res.Scan(&bid, &genre); err != nil {
			panic(err)
		}
		bookgenres[bid] = append(bookgenres[bid], genre)
	}
	res, err = db.Query("SELECT book_id, num_pages, publication_date FROM book")
	if err != nil {
		panic(err)
	}
	books := make(map[int]Book)
	for res.Next() {
		var bid, npages int
		var date time.Time
		if err = res.Scan(&bid, &npages, &date); err != nil {
			panic(err)
		}
		var attributes = map[string]string{"history, historical fiction, biography": "0", "children": "0", "romance": "0", "fantasy, paranormal": "0", "fiction": "0", "mystery, thriller, crime": "0", "poetry": "0", "young-adult": "0", "non-fiction": "0", "comics, graphic": "0", "pages": "0", "age": "0"}
		var pagesone, age string
		if npages > 400 {
			pagesone = "Long"
		} else if npages > 150 {
			pagesone = "Medium"
		} else {
			pagesone = "Short"
		}
		t := time.Date(1900, 1, 1, 0, 0, 0, 0, time.UTC)
		t2 := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
		if date.Before(t) {
			age = "Old"
		} else if date.After(t2) {
			age = "New"
		} else {
			age = "Mid"
		}
		attributes["pages"] = pagesone
		attributes["age"] = age
		for _, value := range bookgenres[bid] {
			attributes[value] = "1"
		}
		books[bid] = Book{
			Attribites: attributes,
		}

	}
	sumOfPrecisions := 0.0
	for _, value := range userrates {
		newbooks := make(map[int]Book)
		for k := range value {
			newbooks[k] = books[k]
		}
		k := len(value) - 3
		z := 0
		testbooks := make(map[int]Book)
		trainbooks := make(map[int]Book)
		testrates := make(map[int]int)
		trainrates := make(map[int]int)
		for key, val := range newbooks {
			if z < k {
				trainbooks[key] = val
				trainrates[key] = value[key]
			} else {
				testbooks[key] = val
				testrates[key] = value[key]
			}
			z++
		}
		root := BuildTree(trainrates, trainbooks, attributesValues)
		precision := Classify(root, testrates, testbooks)
		//log.Printf("user %s got average precision of %f with %d data\n", key, precision, len(value))
		sumOfPrecisions += precision
	}
	log.Printf("avg precision for %d users is %f\n", len(userrates), sumOfPrecisions/float64(len(userrates)))
}
