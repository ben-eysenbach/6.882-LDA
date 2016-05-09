#include "LDA.h"
#define EPS 1.0e-10

// N - number of words
// M - number of documents
// V - size of vocabulary
// phi - M x N x K matrix of floats
//     - For each document, for each word, what is the probability it belongs to each cluster
// gamma - M x K matrix of floats
//       - For each document, what is the probability it belongs to each cluster
// beta - K x V
//      - Probability of each word given each topic

using namespace std;

//Assumes that all documents are stored in a single file, which one document per line. On each line words are separated by spaces
vector<vector<int>> read_documents(string filename, unordered_map<string, int> map) {
    ifstream f(filename);
    string line;
    stringstream line_stream;
    string word;
    char delimiter = ' ';
    // string word;
    // vector<vector<string>> documents;
    // vector<string>;
    vector<vector<int>> documents;
    int count = 0;
    int index;

    while (getline(f, line)) {
        line_stream = stringstream(line);
        vector<int> indices;
        // delimiter must be char type (single quotes)
        while (getline(line_stream, word, delimiter)) {
            if (map.count(word) > 0) {
                index = map[word];
                indices.push_back(index);
            }
        }
        documents.push_back(indices);
    }
    return documents;
}

// Creates a map from word -> index
void construct_dictionary(string filename, int vocab_size, unordered_map<string, int> &words, vector<string> &word_list) {
    ifstream f(filename);
    string w;
    int index = 0;
    while (getline(f, w) && (index < vocab_size)) {
        if (words.count(w) == 0) {
            words[w] = index;
            word_list.push_back(w);
            index++;
        } else {
            // cout << "Duplicate:" << w << endl;
        }
    }
}

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

vector<vector<vector<float>>> initialize_phi(vector<vector<int>> documents, int K) {
    vector<vector<vector<float>>> phi;
    int M = documents.size();
    for (int m = 0; m < M; m++) {
        int N = documents[m].size();
        vector<vector<float>> matrix;
        for (int i = 0; i < N; i++) {
            vector<float> row (K, 1.0 / K);
            for (int j = 0; j < K; j++) {
                row[j] += rand_float();
            }
            matrix.push_back(row);
        }
        phi.push_back(matrix);
    }
    return phi;
}

vector<vector<float>> initialize_gamma(vector<vector<int>> documents, int K, vector<float> alpha) {
    assert(K == alpha.size());
    int M = documents.size();
    vector<vector<float>> gamma;
    for (int m = 0; m < M; m++) {
        int N = documents[m].size();
        vector<float> row (K, (float) N / (float) K);
        for (int i = 0; i < K; i++) {
            row[i] += alpha[i] + rand_float();
        }
        gamma.push_back(row);
    }
    return gamma;
}

//We initialize with Zipf distribution
vector<vector<float>> initialize_beta(int K, int V) {
    vector<vector<float>> beta;

    // // First, compute the normalizing constant
    // float z = 0;
    // for (int i = 0; i < K; i++) {
    //     z += 1.0 / (i + 1);
    // }
    //
    // for (int i = 0; i < K; i++) {
    //     vector<float> row(V);
    //     for (int j = 0; j < V; j++) {
    //         row[j] = 1.0 / (z * (j + 1));
    //     }
    //     beta.push_back(row);
    // }
    for (int i = 0; i < K; i++) {
        vector<float> row(V);
        float sum = 0;
        for (int j = 0; j < V; j++) {
            row[j] = rand_float();
            sum += row[j];
        }

        for (int j = 0; j < V; j++) {
            row[j] /= sum;
        }
        beta.push_back(row);
    }
    return beta;
}

vector<float> initialize_alpha(int K) {
    vector<float> alpha (K, 1.0);
    for (int i = 0; i < K; i++) {
        alpha[i] = rand_float();
    }
    return alpha;
}

// Update phi and gamma using fixed alpha and beta
void E_step(vector<vector<int>> documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha) {
    int K = alpha.size();
    int M = documents.size();
    float tmp;
    // For each document ...
    for (int m = 0; m < M; m++) {
        // Repeat until convergence
        float diff_gamma = 1;
        int iter = 0;
        while ((diff_gamma > EPS) && (iter < 20)) {
            diff_gamma = 0;
            int N = documents[m].size();
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int i = 0; i < K; i++) {
                    int word_index = documents[m][n];
                    // cout << gamma[m][i] << endl;
                    assert(gamma[m][i] > 0);
                    float exponent = boost::math::digamma(gamma[m][i]);
                    phi[m][n][i] = beta[i][word_index] * exp(exponent);
                    sum += phi[m][n][i];
                }
                // normalize phi
                for (int i = 0; i < K; i++) {
                    phi[m][n][i] /= sum;
                }
            }
            for (int i = 0; i < K; i++) {
                float sum = 0;
                for (int n = 0; n < N; n++) {
                    sum += phi[m][n][i];
                }
                tmp = gamma[m][i];
                // cout << alpha[i] << endl;
                gamma[m][i] = alpha[i] + sum;
                diff_gamma += abs(gamma[m][i] - tmp);
            }
            // cout << "Gamma Update: " << diff_gamma << endl;
            iter++;
        }
    }
}

void update_beta(vector<vector<int>> documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &beta) {
    //beta is K x V dimensional
    // first fill b with zeros
    int K = beta.size();
    int V = beta[0].size();
    float eta = 1;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < V; j++) {
            beta[i][j] = eta;
        }
    }

    //Now, increment each entry by phi
    int M = documents.size();
    for (int m = 0; m < M; m++) {
        int N = documents[m].size();
        for (int n = 0; n < N; n++) {
            int word_index = documents[m][n];
            for (int i = 0; i < K; i++) {
                beta[i][word_index] += phi[m][n][i];
            }
        }
    }

    //Finally, normalize each row of beta
    for (int i = 0; i < K; i++) {
        float sum = 0;
        for (int j = 0; j < V; j++) {
            sum += beta[i][j];
        }
        for (int j = 0; j < V; j++) {
            beta[i][j] /= sum;
        }
    }
}

void show_alpha(vector<float> alpha) {
    cout << "Alpha: [";
    for (float entry : alpha) {
        cout << entry << ", ";
    }
    cout << "]" << endl;
}

void update_alpha(vector<float> &alpha, vector<vector<float>> &gamma, float alpha_init = 1.0) {
    int K = alpha.size();
    int M = gamma.size();

    float diff_alpha = 1.0;
    int iter = 0;
    while ((diff_alpha > EPS) && (iter < 20)) {
        // cout << "Update Alpha (iter=" << iter << "):\n\t";
        float sum_alpha = 0;
        for (int i = 0; i < K; i++) {
            sum_alpha += alpha[i];
        }
        // cout << "sum_alpha: " << sum_alpha << endl;
        // cout << "gamma(suma_alpha): " <<

        // compute g
        vector<float> g(K, 0);
        for (int i = 0; i < K; i++) {
            assert(sum_alpha > 0);
            assert(alpha[i] > 0);
            g[i] = M * (boost::math::digamma(sum_alpha) - boost::math::digamma(alpha[i]));
            for (int m = 0; m < M; m++) {
                float sum_gamma = 0;
                for (int k = 0; k < K; k++) {
                    sum_gamma += gamma[m][k];
                }
                assert(gamma[m][i] > 0);
                assert(sum_gamma > 0);
                g[i] += boost::math::digamma(gamma[m][i]) - boost::math::digamma(sum_gamma);
            }
        }

        //compute h
        // float z = -1.0 * boost::math::trigamma(sum_alpha);
        float z = M * boost::math::trigamma(sum_alpha);
        vector<float> h(K, 0);
        for (int i = 0; i < K; i++) {
            h[i] = -M * boost::math::trigamma(alpha[i]);
        }

        //compute c
        float numerator = 0;
        float denominator = 0;
        for (int i = 0; i < K; i++) {
            numerator += g[i] / h[i];
            denominator += 1.0 / h[i];
        }
        denominator += 1.0 / z;
        float c = numerator / denominator;


        diff_alpha = 0;
        for (int i = 0; i < K; i++) {
            alpha[i] -= (g[i] - c) / h[i];
            diff_alpha += abs((g[i] - c) / h[i]);
        }

        // If alpha is negative, restart with a smaller initialization
        for (int i = 0; i < K; i++) {
            if (alpha[i] < EPS) {
                fill(alpha.begin(), alpha.end(), alpha_init);
                update_alpha(alpha, gamma, alpha_init / 10.0);
            }
        }

        // cout << "Alpha Update(" << iter << "): " << diff_alpha << endl;
        iter++;
    }
}

//update alpha and beta using fixed phi and gamma
void M_step(vector<vector<int>> &documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha) {
    update_beta(documents, phi, beta);
    update_alpha(alpha, gamma);
}

void show_beta(vector<vector<float>> beta, vector<string> word_list) {
    int K = beta.size();
    int V = beta[0].size();
    int display = 10;
    for (int i = 0; i < K; i++) {
        cout << "Beta Cluster " << i << ":" << endl;
        for (int iter = 0; iter < display; iter++) {
            float best_beta = 0;
            int best_index = 0;
            for (int j = 0; j < V; j++) {
                if (beta[i][j] > best_beta) {
                    best_beta = beta[i][j];
                    best_index = j;
                }
            }
            cout << "\t" << word_list[best_index] << " : " << best_beta << endl;
            beta[i][best_index] = 0;
        }
    }
    cout << endl;
}

void show_gamma(vector<vector<float>> &gamma) {
    int M = gamma.size();
    int K = gamma[0].size();
    for (int m = 0; m < M; m++) {
        cout << "Gamma Document " << m << ":" << endl;
        for (int i = 0; i < K; i++) {
            cout << "\t" << i << ": " << gamma[m][i] << endl;
        }
    }
    cout << endl;
}

// float prob_gamma(vector<float> &gamma, vector<float> &alpha) {
//     float normalizer_top = 1.0;
//     float normalizer_bottom_sum = 0.0;
//     float prod = 1.0;
//     assert(gamma.size() == alpha.size());
//     int K = alpha.size();
//     for (int k = 0; k < K; k++) {
//         prod *= pow(gamma[k], alpha[k] - 1.0);
//         normalizer_top *= boost::math::tgamma(alpha[k]);
//         normalizer_bottom_sum += alpha[k];
//     }
//     float normalizer_bottom = 0.0;
//     normalizer_bottom = boost::math::tgamma(normalizer_bottom_sum);
//     float normalizer = normalizer_top / normalizer_bottom;
//     float prob = prod / normalizer;
//     return prob;
// }

void show_phi(vector<vector<int>> &documents, vector<string> word_list, vector<vector<vector<float>>> &phi) {
    int M = documents.size();
    int K = phi[0][0].size();
    for (int m = 0; m < M; m++) {
        int N = documents[m].size();
        for (int n = 0; n < N; n++) {
            int word_index = documents[m][n];
            string word = word_list[word_index];
            cout << word << endl;
            for (int k = 0; k < K; k++) {
                cout << "\t" << phi[m][n][k] << endl;
            }
        }
        cout << endl;
    }
    cout << endl;
}

// From pg 1020, Eq 15
float compute_likelihood(vector<vector<int>> &documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha) {
    int M = documents.size();
    int K = alpha.size();
    float likelihood = 0;

    int cutoff = min(M, 100);

    for (int m = 0; m < cutoff; m++) {
        float sum_alpha = 0;
        float sum_gamma = 0;

        // first line
        for (int k = 0; k < K; k++) {
            sum_alpha += alpha[k];
            sum_gamma += gamma[m][k];
            likelihood -= boost::math::lgamma(alpha[k]);
        }
        likelihood += boost::math::lgamma(sum_alpha);
        for (int k = 0; k < K; k++) {
            likelihood += (alpha[k] - 1.0) * (boost::math::digamma(gamma[m][k]) - boost::math::digamma(sum_gamma));
        }

        //second line
        int N = documents[m].size();
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                likelihood += phi[m][n][k] * (boost::math::digamma(gamma[m][k]) - boost::math::digamma(sum_gamma));

                //third line
                int word_index = documents[m][n];
                likelihood += phi[m][n][k] * log(beta[k][word_index]);
            }
        }

        //fourth line
        likelihood -= boost::math::lgamma(sum_gamma);
        for (int k = 0; k < K; k++) {
            likelihood += boost::math::lgamma(gamma[m][k]);
            likelihood -= (gamma[m][k] - 1.0) * (boost::math::digamma(gamma[m][k]) - boost::math::digamma(sum_gamma));
        }

        //fifth line
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                likelihood -= phi[m][n][k] * log(phi[m][n][k]);
            }
        }
    }

    return likelihood;
}

// float perplexity(vector<vector<int>> &documents, vector<vector<vector<float>>> &phi, vector<vector<float>> &gamma, vector<vector<float>> &beta, vector<float> &alpha) {
//     int num_words = 0;
//     int M = documents.size();
//     int K = alpha.size();
//     float sum_prob = 0;
//     for (int m = 0; m < M; m++) {
//         int N = documents[m].size();
//         num_words += N;
//         for (int n = 0; n < N; n++) {
//             float prob = 0;
//             for (int k = 0; k < K; k++) {
//                 float prob_topic = gamma[m][k];
//                 int word_index = documents[m][n];
//                 float prob_word_given_topic = beta[k][word_index];
//                 float prob_word_in_topic = phi[m][n][k];
//                 prob += prob_topic * prob_word_given_topic * prob_word_in_topic;
//             }
//             float prob_mixture = prob_gamma(gamma[m], alpha);
//             prob *= prob_mixture;
//             sum_prob += log(prob);
//         }
//     }
//     float perp = exp(-1.0 * sum_prob / num_words);
//
//     return perp;
// }

void debug_alpha(){
    vector<float> alpha (2, 0.1);

    vector<vector<float>> gamma;
    gamma.push_back({0.9, 0.1});

    update_alpha(alpha, gamma);
}

int main(int argc, char const *argv[]) {
    srand(time(0));
    int K; // number of clusters
    if (argc == 2) {
        K = atoi(argv[1]);
    } else {
        K = 2;
    }
    cout << "Num clusters: " << K << endl;

    // string filename_dictionary = "datasets/nyt_vocab_1000.txt";
    // string filename_dataset = "datasets/nyt_train_1000.txt";

    string filename_dictionary = "datasets/reuters_vocab_1000.txt";
    string filename_dataset = "datasets/reuters_train_1000.txt";

    cout << "Constructing dictionary" << endl;
    int vocab_size = 10000000; // Something large enough that it doesn't matter
    unordered_map<string, int> map;
    vector<string> word_list;
    construct_dictionary(filename_dictionary, vocab_size, map, word_list);

    cout << "Initializing parameters" << endl;
    auto alpha = initialize_alpha(K);
    auto documents = read_documents(filename_dataset, map);
    auto gamma = initialize_gamma(documents, K, alpha);
    auto beta = initialize_beta(K, vocab_size);
    auto phi = initialize_phi(documents, K);


    cout << "Starting Variational Inference" << endl;
    cout << "Iteration -1" << endl;
    float likelihood = compute_likelihood(documents, phi, gamma, beta, alpha);
    cout << "Likelihood: " << likelihood << endl;

    for (int iter = 0; iter < 20; iter++) {
        cout << "Iteration: " << iter << endl;
        E_step(documents, phi, gamma, beta, alpha);
        M_step(documents, phi, gamma, beta, alpha);
        likelihood = compute_likelihood(documents, phi, gamma, beta, alpha);
        cout << "Likelihood: " << likelihood << endl;

        show_beta(beta, word_list);
    }
    // show_phi(documents, word_list, phi);
    // show_gamma(gamma);


    return 0;
}
