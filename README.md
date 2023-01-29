# MSci Thesis Project #

This was a year-long research project in partial fulfillment to my MSci Physics degree, alongside my supervisor, Professor Carlo R. Contaldi, MBE. This project involved building semi-stochastic epidemiological models for SARS-CoV-2, exploring the processes, advantages and limitations of creating models with a varying number of assumptions and parameters, and developing a bespoke Monte Carlo Markov Chain algorithm with adapting proposal functions utilising parameter correlations to efficiently infer parameters, formulating a framework of Bayesian inference. The algorithm was tested against simulated data, and then used to obtain probability distributions for the parameters of our models by inferring from real COVID-19 data. These were finally used to make a short-term prediction about the course of the pandemic.

## Organisation ##
The repository contains:
- A directory with all the main MCMC algorithms developed to infer from simulated and real COVID-19 data `./mcmc/`
- A directory with some preliminary research on vaccination efficacy and herd immunity `./preliminary_experiments/`
- The MSci Thesis report `./msci_project_report.pdf`
