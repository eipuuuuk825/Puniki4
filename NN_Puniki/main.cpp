#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>
#include "NeuralNetwork.hpp"

#include <iterator>

template <typename T>
using vector = std::vector<T>;
using std::cout;
using std::endl;

//ユークリッド距離算出関数
double euclidean_distance(const vector<double> &a, const vector<double> &b)
{
	if (a.size() != b.size()) //同じ長さのベクトルかチェック
		throw std::domain_error("In distance, size of vectors does not match");

	double sum = 0.0;
	for (size_t i = 0; i < a.size(); ++i)
		sum += std::pow(a[i] - b[i], 2);
	return std::sqrt(sum);
}

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
	vector<vector<double>> x_l(x.begin(), x.begin() + 50);
	vector<vector<double>> x_t(x.begin() + 50, x.end());
	vector<vector<double>> t_l(t.begin(), t.begin() + 50);
	vector<vector<double>> t_t(t.begin() + 50, t.end());

	so::NeuralNetwork nn({18, 10, 2}, 0.5);
	{
		//E履歴出力用ファイル作成
		std::ofstream ofs("../data_sample/E.csv");
		if (!ofs)
			throw std::fstream::failure("Cannot create ");

		// nn.prelearning(x_l, 1E-4, -1, "deltaE");

		nn.learning(x_l, t_l, 1E-4, "deltaE", -1, &ofs); //学習
	}
	/*
	for (auto i : t)
	{
		for (auto j : i)
			cout << j << ", ";
		cout << endl;
	}

	//結果のファイルを出力
	std::string file_result = "data/result.txt";
	std::ofstream fout;
	fout.open(file_result.c_str());
	if (!fout.is_open())
	{
		std::cout << "ファイルをオープンできません" << std::endl;
		return -1;
	}

	vector<vector<int>> result_table(t_j[0].size()); //結果の表(クラス数*クラス数)作成
	for (auto &&e : result_table)					 //結果の表初期化
	{
		e.resize(t_j[0].size());
		std::fill(e.begin(), e.end(), 0);
	}
	//判定
	int total = 0;	   //判定数
	int correct_n = 0; //正解数
	for (size_t i = 0; i < x_j.size(); ++i)
	{
		vector<double> output = nn.compute(x_j[i]); //NNで計算
													//NNの出力と一番近い回答点を見つける
													//NNの出力ベクトルと一番近い分類ベクトルを探す
		vector<double> dist_v(t_j[i].size());		//分類ベクトルとの距離のベクトル
		for (size_t j = 0; j < dist_v.size(); ++j)
		{
			//分類ベクトル作成
			vector<double> t_ref(t_j[i].size());
			std::fill(t_ref.begin(), t_ref.end(), 0.0); //初期化
			t_ref[j] = 1.0;								//classificationでは選択肢のj番目はj番目のみが1で他が0の教師ベクトルである

			dist_v[j] = euclidean_distance(output, t_ref); //距離計算
		}
		//この時点でdist_v[j]はj番目の分類ベクトルとの距離となっている
		//最短距離となるのは何番目の分類ベクトルとか、を計算
		int result = std::distance(dist_v.begin(), std::min_element(dist_v.begin(), dist_v.end()));
		//教師ベクトルは何番目の分類であったか  ex){0.0, 0.0, 1.0} => 2
		int answer = std::distance(t_j[i].begin(), std::max_element(t_j[i].begin(), t_j[i].end()));

		//結果の表は{NNの判断した分類*回答の分類}
		++result_table[result][answer]; //結果の表の(result, answer)を1増加

		if (result == answer) //NNの判断した分類==回答の分類 なら
			++correct_n;	  //正解数を1増加
		else
		{ //不正解なら
			//どんなふうに間違ったのかを出力
			fout << "wrong_judgement\t\t";
			std::ostream_iterator<double> oit(fout, ", ");
			fout << "output = ";
			std::copy(output.begin(), output.end(), oit);
			fout << "\tanswer = ";
			std::copy(t_j[i].begin(), t_j[i].end(), oit);
			fout << std::endl;
		}
		++total;
	}

	fout << "\nresult_table\n";
	for (auto &&e : result_table) //結果の表出力
	{
		std::ostream_iterator<double> oit(fout, ", ");
		std::copy(e.begin(), e.end(), oit);
		fout << std::endl;
	}
	//正解率出力
	fout << "\naccuracy rate = " << static_cast<double>(correct_n) / total << '\n';

	return 0;
	*/
}
