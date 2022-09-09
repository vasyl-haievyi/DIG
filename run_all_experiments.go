package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

func runSparsityLoop(tmpl, method, dataset string) {
	for _, sparcity := range strings.Split("0.55 0.6 0.65 0.7 0.75 0.8", " ") {
		re := regexp.MustCompile(`sparsity: 0.[0-9]+`)
		text := re.ReplaceAllString(tmpl, "sparsity: "+sparcity)

		fmt.Println(text)

		os.WriteFile(fmt.Sprintf("benchmarks/xgraph/config/explainers/%s.yaml", strings.TrimSuffix(method, "_edges")), []byte(text), 0x666)

		args := strings.Split(fmt.Sprintf("-m benchmarks.xgraph.%s datasets=%s", method, dataset), " ")
		cmd := exec.Command("/home/basil/miniconda3/envs/mgr2/bin/python", args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if err != nil {
			log.Fatal(err)
		}
	}
}

func main() {
	dataset := "bace"

	for _, method := range []string{"gnn_lrp"} {
		bytes, _ := os.ReadFile(fmt.Sprintf("benchmarks/xgraph/config/explainers/%s.yaml", strings.TrimSuffix(method, "_edges")))
		tmpl := string(bytes)

		if method != "custom_explainer" {
			runSparsityLoop(tmpl, method, dataset)
		} else {
			for _, replace := range []string{"all", "C"} {
				for _, replace_alg := range []string{"atom", "number"} {
					for _, weight_alg := range []string{"signed", "absolute"} {

						config := replace + " + " + replace_alg + " + " + weight_alg
						fmt.Printf("Computing configuration %s \n", config)

						replacement_re := regexp.MustCompile("replace_atoms_with: ('all'|'C')")
						replacement_alg_re := regexp.MustCompile("replace_atom_alg: ('number'|'atom')")
						weight_method_re := regexp.MustCompile("calculate_atom_weight_alg: ('absolute'|'signed')")

						text := replacement_re.ReplaceAllString(tmpl, fmt.Sprintf("replace_atoms_with: '%s'", replace))
						text = replacement_alg_re.ReplaceAllString(text, fmt.Sprintf("replace_atom_alg: '%s'", replace_alg))
						text = weight_method_re.ReplaceAllString(text, fmt.Sprintf("calculate_atom_weight_alg: '%s'", weight_alg))

						runSparsityLoop(text, method, dataset)

						if err := os.RemoveAll(fmt.Sprintf("/home/basil/University/UJ/PracaMgr/DIG/benchmarks/xgraph/results/%s/gcn/Custom", dataset)); err != nil {
							log.Fatal(err)
						}
					}
				}
			}
		}

	}
}
