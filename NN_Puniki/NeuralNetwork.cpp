#include "NeuralNetwork.hpp"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>

namespace
{
	/* 例外処理用 */
	void my_exception(const char *func, const std::string msg)
	{
		std::stringstream sstr;
		sstr << "In [" << func << "], " << msg;
		throw std::runtime_error(sstr.str());
	}

	//シグモイド関数
	double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

	/* 渡された vector 等をファイルに出力する */
	template <typename Seq>
	void output_sequence(const Seq &seq, const std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs)
			my_exception(__FNAME__, "cannot open [" + path + "]");

		for (auto &&e : seq)
			ofs << e << std::endl;
	}

	/* 学習済みのパラメータをファイルに出力 */
	void output_learned_param(std::vector<size_t> &n,
							  double eta,
							  std::vector<std::vector<std::vector<double>>> &w,
							  const std::string path_learned_param)
	{
		std::ofstream ofs(path_learned_param);
		if (!ofs)
			my_exception(__FNAME__, "cannot open [" + path_learned_param + "]");

		const char separator = ',';

		/* n */
		for (auto &&i : n)
			ofs << i << separator;
		/* eta */
		ofs << "\n"
			<< eta << std::endl;
		/* w */
		for (auto &&i : w)
		{
			for (auto &&j : i)
			{
				for (auto &&k : j)
					ofs << k << separator;
				ofs << std::endl;
			}
		}
	}
} // namespace

/* str を del で分割する */
std::vector<double> so::split(std::string str, char del)
{
	int first = 0;
	int last = str.find_first_of(del);
	std::vector<double> result;

	while (first < str.size())
	{
		std::string sub_str(str, first, last - first);

		result.push_back(std::stod(sub_str));

		first = last + 1;
		last = str.find_first_of(del, first);

		if (last == std::string::npos)
			last = str.size();
	}
	return result;
}

/* 重み、逆伝播する誤差、各ニューロンの出力の配列を生成 */
so::NeuralNetwork::NeuralNetwork(const vector<size_t> &n_, double eta_, const std::string nn_mode_)
	: n(n_), L(n.size()), eta(eta_), nn_mode(nn_mode_)
{
	init();
}

/* 学習済みの重みを使用する */
so::NeuralNetwork::NeuralNetwork(const std::string nn_mode_, const std::string path_learned_param)
	: nn_mode(nn_mode_)
{
	std::ifstream ifs(path_learned_param);
	if (!ifs)
		my_exception(__FNAME__, "cannot open [" + path_learned_param + "]");
	std::string line;

	/* n */
	std::getline(ifs, line);
	vector<double> splitted = split(line, ',');
	n.resize(splitted.size());
	for (size_t i = 0; i < splitted.size(); ++i)
		n[i] = (size_t)splitted[i];

	/* L */
	L = n.size();

	/* eta */
	std::getline(ifs, line);
	eta = std::stod(line);

	/* メンバ変数の初期化 */
	init();

	/* w */
	for (auto &&i : w)
	{
		for (auto &&j : i)
		{
			std::getline(ifs, line);
			j = split(line, ',');
		}
	}
}

void so::NeuralNetwork::init()
{
	/* nn_mode のエラーチェック */
	if (nn_mode != MODE_C and nn_mode != MODE_R)
		my_exception(__FNAME__, "invalid nn_mode [" + nn_mode + "]");

	// z = vector<vector<double>>
	z.resize(L);
	for (size_t l = 0; l < L; ++l)
	{
		z[l].resize(n[l] + 1); //バイアスの分を余分に確保
		z[l][0] = 1.0;		   //バイアスニューロンの出力は常に１
	}

	d.resize(L);
	for (size_t l = 1; l < L; ++l)
	{ //d[0][i]は入力層だから誤差計算は必要ない
		d[l].resize(n[l] + 1);
		d[l][0] = 0.0; //バイアスは誤差計算に関係ない（初期化不要？）
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
}

std::vector<double> so::NeuralNetwork::compute(const vector<double> &x)
{
	if (x.size() != n[0])
	{
		std::stringstream sstr;
		sstr << "NeuralNetwork::compute\n"
			 << "入力データの次元が不適切です．\n"
			 << "セットされたパラメータの次元：" << n[0]
			 << ", 入力されたデータの次元：" << x.size();
		my_exception(__FNAME__, sstr.str());
	}

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
			if (nn_mode == MODE_R and l == L - 1)
				z[l][k] = sum; // 出力層だけ恒等写像
			else
				z[l][k] = sigmoid(sum);
		}
	}
	vector<double> output(z[L - 1].begin() + 1, z[L - 1].end()); //最終層の出力を格納	// TODO:.back() を使って簡略化
	return std::move(output);
}

void so::NeuralNetwork::back_propagation(const vector<double> &t)
{
	//出力層について計算、重みを更新
	for (size_t j = 1; j < n[L - 1] + 1; ++j)
	{
		if (nn_mode == MODE_C)
			d[L - 1][j] = -(t[j - 1] - z[L - 1][j]) * z[L - 1][j] * (1.0 - z[L - 1][j]);
		else
			d[L - 1][j] = -(t[j - 1] - z[L - 1][j]); /* 回帰分析では出力層の活性化関数として恒等関数を用いるため */
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
			for (size_t i = 0; i < n[l - 1] + 1; ++i)
			{
				w[l - 1][j][i] -= eta * d[l][j] * z[l - 1][i];
			}
		}
	}
}

void so::NeuralNetwork::autoencoder(const vector<vector<double>> &x_v, int l, double epsilon, int limit, const std::string convergence_mode)
{
	/* nn_mode のエラーチェック */
	if (nn_mode == MODE_R)
		my_exception(__FNAME__, "回帰モードでの動作は未確認です");

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
	/* nn_mode のエラーチェック */
	if (nn_mode == MODE_R)
		my_exception(__FNAME__, "回帰モードでの動作は未確認です");

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

void so::NeuralNetwork::prelearning(const vector<vector<double>> &x_v, double epsilon, int limit, const std::string convergence_mode)
{
	/* nn_mode のエラーチェック */
	if (nn_mode == MODE_R)
		my_exception(__FNAME__, "回帰モードでの事前学習の動作は未確認です");

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
	for (size_t j = 1; j < n[L - 1] + 1; ++j)
		sum += std::pow((t[j - 1] - z[L - 1][j]), 2); //教師ベクトルの添え字は0から始まる。一方出力の0番目はバイアス項となっている
	return 0.5 * sum;
}

double so::NeuralNetwork::learning(const vector<vector<double>> &x_v,
								   const vector<vector<double>> &t_v,
								   const std::string path_E_his,
								   const std::string path_learned_param,
								   const std::string convergence_mode,
								   double epsilon,
								   int limit)
{

	vector<double> E_his;
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
		E_his.push_back(E); //エラー履歴に保存
		++count;

		/* 収束判定 */
		if (convergence_mode == "deltaE") //エラーの変化率で判定する方法
		{
			if (count != 1)
			{
				double deltaE = std::abs(E_his[E_his.size() - 1] - E_his[E_his.size() - 2]);
				if (deltaE < epsilon)
					break;
			}
		}
		else if (E < epsilon) //エラーの大きさで判定する方法
			break;
		if (count == limit) //試行回数で打ち切り
			break;
	}

	/* エラー履歴をファイルに出力 */
	if (path_E_his != "")
		output_sequence(E_his, path_E_his);

	/* 学習済みのパラメータをファイルに出力 */
	if (path_learned_param != "")
		output_learned_param(n, eta, w, path_learned_param);

	return E_his.back();
}