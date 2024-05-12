# Improvements-of-LCS-approximation

"This is the code accompanying the paper titled: '**Longest Common Subsequence: Tabular vs. Closed-Form Equation Computation of Subsequence Probability**.'

To run the code, follow the instructions below:
1. Specify the path of the dataset for which you want to obtain the LCS in line 196 of the code.
2. Call the function '**main**' with the inputs: number of strings, number of alphabets, alphabet set, dataset name, dataset type, beta, and flag.
3. The output will be the length of the obtained LCS along with the runtime.

Note: The dataset type can be either '*correlated*' or '*uncorrelated*'. If your dataset is generated randomly, please choose '*uncorrelated*'. Otherwise, if your dataset has some relevance between its sequences, please choose '*correlated*'. If you are unsure of the dataset type, you can determine it using the method introduced in the paper (https://www.sciencedirect.com/science/article/abs/pii/S1476927123000737). The parameter beta is the beam width in the beam search algorithm; you can select its value based on the description provided in the paper."

### citation

```bash
@article{abdi2022longest,
  title={Longest common subsequence: Tabular vs. closed-form equation computation of subsequence probability},
  author={Abdi, Alireza and Hooshmand, Mohsen},
  journal={arXiv preprint arXiv:2206.11726},
  year={2022}
}
```
