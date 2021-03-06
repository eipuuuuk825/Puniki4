#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include "NeuralNetwork.hpp"

template <typename T>
using vector = std::vector<T>;
using std::cout;
using std::endl;

int main(void)
{
	/*-----------------------------------------------
	*
	* パラメータ
	*
	-----------------------------------------------*/
	size_t input_data_dim = 18; /* 入力データの次元 */
	const vector<size_t> neural_num{18, 10, 2};
	// const vector<size_t> neural_num{12, 7, 2};
	// const vector<size_t> neural_num{9, 6, 2};
	const double eta = 0.05;

	/*-----------------------------------------------
	*
	* データセットを読み込む
	*
	-----------------------------------------------*/
	/* データセットのフォーマットが変更されているため，
	このプログラムも変更が必要 */
	std::ifstream ifs("data_set.csv");
	if (!ifs)
		throw std::fstream::failure("Cannot open");
	std::string line;

	/* 教師データの平均，標準分散 */
	double xt_mean;
	double xt_std;
	double tt_mean;
	double tt_std;
	double yt;

	/* yt */
	std::getline(ifs, line);
	std::getline(ifs, line);
	yt = so::split(line, ',')[0];

	/* mean */
	std::getline(ifs, line);
	std::getline(ifs, line);
	{
		vector<double> tmp = so::split(line.substr(0, line.size() - 1), ',');
		xt_mean = tmp[input_data_dim];
		tt_mean = tmp[input_data_dim + 1];
	}

	/* std */
	std::getline(ifs, line);
	std::getline(ifs, line);
	{
		vector<double> tmp = so::split(line.substr(0, line.size() - 1), ',');
		xt_std = tmp[input_data_dim];
		tt_std = tmp[input_data_dim + 1];
	}

	/* データセット */
	vector<vector<double>> data_set;
	std::getline(ifs, line);
	while (std::getline(ifs, line))
		data_set.emplace_back(so::split(line.substr(0, line.size() - 1), ','));

	/*-----------------------------------------------
	*
	* 入力データと教師データを用意
	*
	-----------------------------------------------*/
	vector<vector<double>> x, t;
	for (auto &&e : data_set)
	{
		vector<double> tmp_x(e.begin(), e.begin() + input_data_dim);
		vector<double> tmp_t(e.begin() + input_data_dim, e.end());
		x.push_back(std::move(tmp_x));
		t.push_back(std::move(tmp_t));
	}

	/*-----------------------------------------------
	*
	* 学習データとテストデータに分ける
	*
	-----------------------------------------------*/
	/* 前方50で学習 */
	vector<vector<double>> xl(x.begin(), x.begin() + 50);
	vector<vector<double>> xt(x.begin() + 50, x.end());
	vector<vector<double>> tl(t.begin(), t.begin() + 50);
	vector<vector<double>> tt(t.begin() + 50, t.end());
	/* 後方50で学習 */
	// vector<vector<double>> xt(x.begin(), x.begin() + 50);
	// vector<vector<double>> xl(x.begin() + 50, x.end());
	// vector<vector<double>> tt(t.begin(), t.begin() + 50);
	// vector<vector<double>> tl(t.begin() + 50, t.end());

	/*-----------------------------------------------
	*
	* 学習
	*
	-----------------------------------------------*/
	so::NeuralNetwork nn(neural_num, eta, "regression");
	nn.learning(xl, tl, "E.csv", "learned_param.csv");

	// so::NeuralNetwork nn("regression", "learned_param.csv");

	/*-----------------------------------------------
	*
	* テストデータで精度を検証
	*
	-----------------------------------------------*/
	auto unnormalize = [](double mean, double std, double val) {
		return val * std + mean;
	};

	/* 結果を出力するファイル */
	std::ofstream ofs_res("result.csv");
	if (!ofs_res)
		throw std::fstream::failure("Cannot create ");
	for (size_t i = 0; i < xt.size(); ++i)
	{
		vector<double> output = nn.compute(xt[i]); //NNで計算

		ofs_res << tt[i][0] << ", " << tt[i][1] << ", "
				<< output[0] << ", " << output[1] << endl;

		// double truex = unnormalize(xt_mean, xt_std, tt[i][0]);
		// double truet = unnormalize(tt_mean, tt_std, tt[i][1]);
		// double predictx = unnormalize(xt_mean, xt_std, output[0]);
		// double predictt = unnormalize(tt_mean, tt_std, output[1]);
		// double dx = predictx - truex;
		// double dt = predictt - truet;
		// ofs_res << truex << ", " << truet << ", "
		// 		<< predictx << ", " << predictt << endl;
	}

	return 0;
}
