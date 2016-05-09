#include <iostream>
#include <set>
#include <string>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <istream>
#include <vector>
#include <assert.h>
#include <boost/math/special_functions/gamma.hpp> // gamma
#include <boost/math/special_functions/digamma.hpp> // derivative of log gamma
#include <boost/math/special_functions/trigamma.hpp> // derivative of digamma
#include <math.h>
#include <cmath>
#include <algorithm>

using namespace std;

void construct_dictionary(string filename, int vocab_size, unordered_map<string, int> &words, vector<string> &word_list);
void E_step(vector<vector<int>> documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha);
vector<float> initialize_alpha(int K);
vector<vector<float>> initialize_beta(int K, int V);
vector<vector<float>> initialize_gamma(vector<vector<int>> documents, int K, vector<float> alpha);
vector<vector<vector<float>>> initialize_phi(vector<vector<int>> documents, int K);
void M_step(vector<vector<int>> &documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha);
vector<vector<int>> read_documents(string filename, unordered_map<string, int> map);
float rand_float();
void show_alpha(vector<float> alpha);
void show_beta(vector<vector<float>> beta, vector<string> word_list);
void show_gamma(vector<vector<float>> &gamma);
void update_alpha(vector<float> &alpha, vector<vector<float>> &gamma, float alpha_init);
void update_beta(vector<vector<int>> documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &beta);
float perplexity(vector<vector<int>> &documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha);
