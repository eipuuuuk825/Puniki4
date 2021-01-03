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
	enum
	{
		DATA_XT_6 = 12,
		DATA_XYT_6 = 18,
		DATA_XYT_12 = 36
	};
	size_t input_data_dim = DATA_XYT_6; /* 入力データの次元 */
	// const vector<size_t> neural_num{12, 7, 2};
	const vector<size_t> neural_num{18, 10, 2};

	/*-----------------------------------------------
	*
	* サンプルデータを読み込む
	*
	-----------------------------------------------*/
	vector<vector<double>> table;
	std::ifstream ifs("../data_sample/1.csv");
	if (!ifs)
		throw std::fstream::failure("Cannot open");
	std::string line;

	/* 教師データの平均，標準分散 */
	double xt_mean = 0;
	double xt_std = 0;
	double tt_mean = 0;
	double tt_std = 0;
	std::getline(ifs, line);
	{
		std::istringstream line_s(line);
		vector<double> line_v;
		std::string temp;
		while (std::getline(line_s, temp, ' '))
			line_v.push_back(std::stod(temp));
		xt_mean = line_v[0];
		xt_std = line_v[1];
		tt_mean = line_v[2];
		tt_std = line_v[3];
	}

	/* サンプルデータ */
	while (std::getline(ifs, line))
	{
		std::istringstream line_s(line);
		vector<double> line_v;
		std::string temp;
		while (std::getline(line_s, temp, ' '))
			line_v.push_back(std::stod(temp));
		table.push_back(std::move(line_v));
	}

	/*-----------------------------------------------
	*
	* 入力データと教師データを用意
	*
	-----------------------------------------------*/
	vector<vector<double>> x, t;
	for (auto &&e : table)
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

	so::NeuralNetwork nn(neural_num, 0.5);
	{
		//E履歴出力用ファイル作成
		std::ofstream ofs("E.csv");
		if (!ofs)
			throw std::fstream::failure("Cannot create ");

		// nn.prelearning(xl, 1E-4, -1, "deltaE");

		nn.learning(xl, tl, 1E-4, "deltaE", -1, &ofs); //学習
	}

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
