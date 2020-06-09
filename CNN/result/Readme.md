## results

When testing our auto-HPO system, the results of the experimental group and the control groups will be saved here. Results includes: 1. classification accuracy, 2.time consumption, 3.test loss of the CNN

### Hyper-parameter output

- predicted_parameters: optimized hyper-parameters predicted by the trained MDPN
- predicted_parameters_random: hyper-parameters predicted by the random MDPN (control group)

### Performance of CNN using hyper-parameters above

- results_mdpn: accuracy, time, loss of CNN running with hyper-parameters predicted by MDPN
- results_random_mdpn: accuracy, time, loss of CNN running with hyper-parameters predicted by random MDPN (control group)