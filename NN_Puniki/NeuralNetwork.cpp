#include "NeuralNetwork.hpp"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <functional>
#include <iostream>
#include <ostream>
#include <ios>
#include <stdexcept>
#include <string>

using std::cout;
using std::endl;

//シグモイド関数
namespace
{
	double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
} // namespace

//エラーEの履歴のファイル出力用関数の定義
template <typename Seq>
void output_sequence(const Seq &seq, std::ostream *os)
{
	if (os != nullptr)
		for (auto &&e : seq)
			*os << e << std::endl;
}

//NeuralNetworkのコンストラクタ
//重み、逆伝播する誤差、各ニューロンの出力の配列を生成
so::NeuralNetwork::NeuralNetwork(const vector<size_t> &n_, double eta_)
	: n(n_), L(n.size()), eta(eta_)
{
	z.resize(L);
	for (size_t l = 0; l < L; ++l)
	{
		z[l].resize(n[l] + 1); //バイアスを出力するニューロンを余分に確保
		z[l][0] = 1.0;		   //バイアスニューロンの出力を設定
	}
	d.resize(L);
	for (size_t l = 1; l < L; ++l)
	{						   //d[0][i]は入力層だから誤差計算は必要ない
		d[l].resize(n[l] + 1); //逆伝播する誤差を各層の各ニューロンに用意	添え字に注意
		d[l][0] = 0.0;		   //各層の0番目のニューロンからは誤差が伝播しない
	}
	w.resize(L - 1); //L層あるから層間の数はL-1
	for (size_t l = 0; l < L - 1; ++l)
	{
		w[l].resize(n[l + 1] + 1); //第l層から第l+1層への重みを確保　さらにバイアスニューロンからの重みも余分に確保
		for (size_t k = 0; k < n[l + 1] + 1; ++k)
		{
			w[l][k].resize(n[l] + 1); //添え字に注意 真ん中の添え字が第l+1層のニューロンで最後の添え字が第l層のニューロン
			for (size_t i = 0; i < n[l] + 1; ++i)
			{
				w[l][0][i] = 0.0; //l+1層目のバイアスニューロンへは出力しない　w[l]はそれぞれ行列になる
			}
		}
	}
	reset();
}

void so::NeuralNetwork::reset()
{
	//メルセンヌツイスタを乱数発生器で初期化
	//-1.0～1.0の一様分布を使用
	std::random_device rnd;
	std::array<std::random_device::result_type, std::mt19937::state_size> v;
	std::generate(v.begin(), v.end(), std::ref(rnd));
	std::seed_seq seed(v.begin(), v.end());
	std::mt19937 engine(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	for (size_t l = 0; l < L - 1; ++l)
		for (size_t k = 1; k < n[l + 1] + 1; ++k)
			for (size_t i = 0; i < n[l] + 1; ++i)
				w[l][k][i] = dist(engine); //初期化

	// vector<vector<vector<double>>> my_w;
	// vector<vector<double>> tmp1 = {
	// 	{0, 0, 0},
	// 	{0.557229, -0.18088, -0.902298},
	// 	{-0.156361, 0.283648, 0.219721}};
	// my_w.emplace_back(tmp1);
	// vector<vector<double>> tmp2 = {
	// 	{0, 0, 0},
	// 	{-0.640767, -0.526888, -0.728733},
	// 	{-0.965611, -0.0015774, -0.899652}};
	// my_w.emplace_back(tmp2);
	// w = my_w;

	/* debug */
	// cout << "*** w ***" << endl;
	// for (auto i : w)
	// {
	// 	for (auto j : i)
	// 	{
	// 		for (auto k : j)
	// 			cout << k << ", ";
	// 		cout << endl;
	// 	}
	// 	cout << "----------------" << endl;
	// }
}

std::vector<double> so::NeuralNetwork::compute(const vector<double> &x)
{

	for (size_t i = 1; i < n[0] + 1; ++i)
	{
		z[0][i] = x[i - 1]; //第0層へ入力ベクトルをセット
	}
	for (size_t l = 1; l < L; ++l)
	{ //順方向計算 第l層のk番目のニューロンの出力を計算
		for (size_t k = 1; k < n[l] + 1; ++k)
		{
			double sum = 0.0;
			for (size_t i = 0; i < n[l - 1] + 1; ++i)
			{
				sum += z[l - 1][i] * w[l - 1][k][i];
			}

			/* 出力の計算 */
			if (l == L - 1)
				z[l][k] = sum; // 出力層だけ恒等写像
			else
				z[l][k] = sigmoid(sum);
		}
	}
	vector<double> output(z[L - 1].begin() + 1, z[L - 1].end()); //最終層の出力を格納

	/* debug */
	// cout << "*** z ***" << endl;
	// for (auto i : z)
	// {
	// 	for (auto j : i)
	// 		cout << j << ", ";
	// 	cout << endl;
	// }

	return std::move(output);
}

void so::NeuralNetwork::back_propagation(const vector<double> &t)
{
	//出力層について計算、重みを更新
	for (size_t j = 1; j < n[L - 1] + 1; ++j)
	{
		// d[L - 1][j] = -(t[j - 1] - z[L - 1][j]) * z[L - 1][j] * (1.0 - z[L - 1][j]);
		d[L - 1][j] = -(t[j - 1] - z[L - 1][j]);
		for (size_t i = 0; i < n[L - 2] + 1; ++i)
		{
			w[L - 2][j][i] -= eta * d[L - 1][j] * z[L - 2][i];
		}
	}
	//入力層から出力層の手前までの重みを更新
	for (size_t l = L - 2; l != 0; --l)
	{ //第0層の誤差は計算しない
		for (size_t j = 1; j < n[l] + 1; ++j)
		{
			double sum = 0;
			for (size_t k = 1; k < n[l + 1] + 1; ++k)
			{
				sum += d[l + 1][k] * w[l][k][j];
			}
			d[l][j] = sum * z[l][j] * (1 - z[l][j]);
			// d[l][j] = sum;
			for (size_t i = 0; i < n[l - 1] + 1; ++i)
			{
				w[l - 1][j][i] -= eta * d[l][j] * z[l - 1][i];
			}
		}
	}

	/* debug */
	// cout << "*** d ***" << endl;
	// for (auto i : d)
	// {
	// 	for (auto j : i)
	// 		cout << j << ", ";
	// 	cout << endl;
	// }
	// cout << "*** w ***" << endl;
	// for (auto i : w)
	// {
	// 	for (auto j : i)
	// 	{
	// 		for (auto k : j)
	// 			cout << k << ", ";
	// 		cout << endl;
	// 	}
	// 	cout << "----------------" << endl;
	// }
}

void so::NeuralNetwork::autoencoder(const vector<vector<double>> &x_v, int l, double epsilon, int limit, const std::string &convergence_mode)
{
	//自己符号化器を作成する
	//第l-1層と第l層を訓練する
	//l層からl_decode層は恒等写像とする
	//decoderのためのニューロン出力とデルタのベクトルを確保
	std::vector<double> z_ae;
	z_ae.resize(n[l - 1] + 1);
	z_ae[0] = 1.0;
	std::vector<double> d_ae;
	d_ae.resize(n[l - 1] + 1);
	d_ae[0] = 0.0;

	double E = 0.0;
	double E_old = 0.0;
	int count = 0;

	while (true)
	{
		E_old = E;
		E = 0.0;
		for (size_t p = 0; p < x_v.size(); ++p)
		{
			//入力ベクトルをセット
			for (size_t i = 1; i < n[l - 1] + 1; ++i)
			{
				z[l - 1][i] = x_v[p][i - 1];
			}
			//まず自己符号化器のencodeを計算する
			for (size_t i = 1; i < n[l] + 1; ++i)
			{
				double sum = 0.0;
				for (size_t h = 0; h < n[l - 1] + 1; ++h)
				{
					sum += z[l - 1][h] * w[l - 1][i][h];
				}
				z[l][i] = sigmoid(sum);
			}
			//次に自己符号化器のdecodeを計算する
			for (size_t j = 1; j < n[l - 1] + 1; ++j)
			{
				z_ae[j] = 0.0;
				for (size_t i = 1; i < n[l] + 1; ++i)
				{
					z_ae[j] += z[l][i] * w[l - 1][i][j];
				}
			}
			//各入力に対して誤差逆伝播法で重みを更新する
			//出力層でのデルタd_aeは次のようになる
			for (size_t j = 1; j < n[l - 1] + 1; ++j)
			{
				d_ae[j] = z_ae[j] - z[l - 1][j];
			}
			//重み更新　重み共有をすることで更新するのはdecoder層だけでよい
			for (size_t j = 1; j < n[l - 1] + 1; ++j)
			{
				for (size_t i = 0; i < n[l] + 1; ++i)
				{
					w[l - 1][i][j] -= eta * d_ae[j] * z[l][i];
				}
				//誤差関数を出力
				E += d_ae[j] * d_ae[j];
			}
		}
		++count;
		if (convergence_mode == "deltaE")
		{
			if (std::abs(E - E_old) < epsilon)
				break;
		}
		else
		{
			if (E < epsilon)
				break;
		}

		if (count == limit)
			break;
	}
}

std::vector<double> so::NeuralNetwork::compute_lth_layer_output(const vector<double> &x, int u)
{

	for (size_t i = 1; i < n[0] + 1; ++i)
	{
		z[0][i] = x[i - 1]; //第0層へ入力ベクトルをセット
	}

	for (int l = 1; l < u + 1; ++l)
	{ //順方向計算 第l層のk番目のニューロンの出力を計算
		for (size_t k = 1; k < n[l] + 1; ++k)
		{
			double sum = 0.0;
			for (size_t i = 0; i < n[l - 1] + 1; ++i)
			{
				sum += z[l - 1][i] * w[l - 1][k][i];
			}
			z[l][k] = sigmoid(sum);
		}
	}
	vector<double> output(z[u].begin() + 1, z[u].end()); //最終層の出力を格納
	return std::move(output);
}

void so::NeuralNetwork::prelearning(const vector<vector<double>> &x_v, double epsilon, int limit, const std::string &convergence_mode)
{
	int p = x_v.size();
	vector<vector<double>> output;
	output.resize(p);

	for (int k = 0; k < p; ++k)
	{
		output[k].resize(x_v[0].size());
		for (size_t j = 0; j < x_v[0].size(); ++j)
		{
			output[k][j] = x_v[k][j];
		}
	}
	for (size_t l = 1; l < L - 1; ++l)
	{
		autoencoder(output, l, epsilon, limit, convergence_mode);
		for (int k = 0; k < p; ++k)
		{
			output[k].clear();
			output[k].resize(n[l]);
			vector<double> temp = compute_lth_layer_output(x_v[k], l);
			for (size_t j = 0; j < temp.size(); j++)
			{
				output[k][j] = temp[j];
			}
		}
	}
}

double so::NeuralNetwork::error(const vector<double> &t) const
{
	double sum = 0.0;
	for (size_t j = 1; j <= n[L - 1]; ++j)
		sum += std::pow((t[j - 1] - z[L - 1][j]), 2); //教師ベクトルの添え字は0から始まる。一方出力の0番目はバイアス項となっている
	/* debug */
	// cout << "*** error ***\n"
	// 	 << 0.5 * sum << endl;
	return 0.5 * sum;
}

double so::NeuralNetwork::learning(const vector<vector<double>> &x_v,
								   const vector<vector<double>> &t_v,
								   double epsilon,
								   const std::string &convergence_mode,
								   int limit,
								   std::ostream *os)
{

	vector<double> E_v;
	int count = 0; // 試行回数カウンタ
	while (true)
	{
		double E = 0.0;
		for (size_t p = 0; p < x_v.size(); ++p)
		{							  //p番目の入力ベクトルと入力層のニューロン数を比較し違ったらエラー
			compute(x_v[p]);		  //順方向計算
			back_propagation(t_v[p]); //逆誤差伝播計算
			E += error(t_v[p]);		  //エラーを計算
		}
		E_v.push_back(E); //エラー履歴に保存
		++count;

		/* 収束判定 */
		if (convergence_mode == "deltaE") //エラーの変化率で判定する方法
		{
			if (count != 1)
			{
				double deltaE = std::abs(E_v[E_v.size() - 1] - E_v[E_v.size() - 2]);
				if (deltaE < epsilon)
					break;
			}
		}
		else if (E < epsilon) //エラーの大きさで判定する方法
			break;
		if (count == limit) //試行回数で打ち切り
			break;
	}
	output_sequence(E_v, os); //エラー履歴をファイルに出力
	return E_v.back();
}
