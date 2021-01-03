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

template<typename T>
using vector = std::vector<T>;

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

int main(int argc, char **argv)
{
	std::string filename = "data/config_Banknote.txt";
	std::string layer_s;
	std::string eta_s;
	std::string epsilon_s;
	std::string limit_s;
	std::string convergence_mode;
	std::string attribute_s;
	std::string element_s;
	std::string temp;
	vector<size_t> neural_number_vector;


	//読み込み用ファイルストリーム
	std::ifstream fin;

	/*ファイルから読み込む*/
	fin.open(filename.c_str());
	//読み込めなければエラー
	if (!fin.is_open()) {
		std::cout << "ファイルを読み込めませんでした。" << std::endl;
		return -2;
	}
	//読み込んだ文字列をstrへ代入

	std::getline(fin,layer_s);
	std::getline(fin, eta_s);
	std::getline(fin, epsilon_s);
	std::getline(fin, limit_s);
	std::getline(fin, convergence_mode);


	fin.close();
	
	//層ごとのニューロン数のベクトルについて、文字列からsize_tのvectorに変換
	{
		std::istringstream s(layer_s);
		while (std::getline(s, temp, ' '))
			neural_number_vector.push_back(std::stoi(temp));
	}

	double eta;
	double epsilon;
	int limit;

	//読み込んだ値をそれぞれdouble,intに変換
	std::istringstream eta_ist(eta_s);
	std::getline(eta_ist, temp, '\n');
	eta = std::stod(temp);
	std::istringstream epsilon_ist(epsilon_s);
	std::getline(epsilon_ist, temp, '\n');
	epsilon = std::stod(temp);
	std::istringstream limit_ist(limit_s);
	std::getline(limit_ist, temp, '\n');
	limit = std::stoi(temp);

	//ファイルから入力_学習,入力_判定,教師_学習,教師_判定ベクトルベクトルを読み込む

		vector<vector<double>> table;
		std::ifstream ifs("data/Banknote.txt");
		if (!ifs)
			throw std::fstream::failure("Cannot open");
		std::string line;
		while(std::getline(ifs, line)) {
			std::istringstream line_s(line);
			vector<double>line_v;
			std::string temp;
			while (std::getline(line_s, temp, ' '))
				line_v.push_back(std::stod(temp));
			table.push_back(std::move(line_v));
		}
		//正規化
		std::array<double, 4>average_v;
		for (auto&&e : table) {
			for (size_t i = 0; i < 4; ++i) {
				average_v[i] += e[i];
			}
		}
		int N = table.size();
		for (size_t i = 0; i < 4; ++i) {
			average_v[i] = average_v[i] / N;
		}
		std::array<double, 4>sum_v;
		for (auto &&e : table) {
			for (size_t i = 0; i < 4; ++i) {
				e[i] = e[i] - average_v[i];
				sum_v[i] += e[i] * e[i];
			}
		}
		std::array<double, 4>sigma_v;
		for (size_t i = 0; i < 4; ++i) {
			sigma_v[i] = std::sqrt(sum_v[i] / N);
		}
		for (auto &&e : table) {
			for (size_t i = 0; i < 4; ++i) {
				e[i] = e[i] / sigma_v[i];
			}
		}
		vector<vector<double>> x;
		for (auto &&e : table) {
			vector<double> temp(4);
			std::copy(e.begin() , e.end()-1, temp.begin());
			x.push_back(std::move(temp));
		}
		vector<vector<double>> t;
		for (auto &&e : table) {
			vector<double>temp(2);
			double species = e[4];
			if (species == 0.0)
				temp = { 1.0,0.0 };
			else if (species == 1.0)
				temp = { 0.0,1.0 };
			else
				throw std::runtime_error("In Banknote,species did not match 1.0 or 0.0");

			t.push_back(std::move(temp));
		}

		//学習と判定に分離
		/*
		//前方50で学習
		vector<vector<double> > x_s(x.begin(), x.begin() + 50);
		vector<vector<double> > x_j(x.begin() + 50, x.end());
		vector<vector<double> > t_s(t.begin(), t.begin() + 50);
		vector<vector<double> > t_j(t.begin() + 50, t.end());
		*/

		//中間50で学習
		vector<vector<double> > x_s(x.begin() , x.begin()+150);
		vector<vector<double> > x_j(x.begin()+150, x.end() -150);
		vector<vector<double> > t_s(t.begin() , t.begin() + 150);
		vector<vector<double> > t_j(t.begin()+150, t.end()-150);
		x_s.insert(x_s.end(), x.end()-150, x.end());
		t_s.insert(t_s.end(), t.end()-150, t.end());

		/*
		//後方50で学習
		vector<vector<double> > x_s(x.end() - 50, x.end());
		vector<vector<double> > x_j(x.begin(), x.end() - 50);
		vector<vector<double> > t_s(t.end() - 50, t.end());
		vector<vector<double> > t_j(t.begin(), t.end() - 50);
		*/

	so::NeuralNetwork nn(neural_number_vector, eta); //NN作成
	{
		//E履歴出力用ファイル作成
		std::ofstream ofs("data/E_Banknote.csv");
		if (!ofs)
			throw std::fstream::failure("Cannot create ");

		nn.prelearning(x_s, epsilon, limit,convergence_mode);

		nn.learning(x_s, t_s, epsilon, convergence_mode, limit, &ofs); //学習
	}

	//結果のファイルを出力
	std::string file_result = "data/result_Banknote.txt";
	std::ofstream fout;
	fout.open(file_result.c_str());
	if (!fout.is_open()) {
		std::cout << "ファイルをオープンできません" << std::endl;
		return -1;
	}

	vector<vector<int> > result_table(t_j[0].size()); //結果の表(クラス数*クラス数)作成
	for (auto &&e : result_table) //結果の表初期化
	{
		e.resize(t_j[0].size());
		std::fill(e.begin(), e.end(), 0);
	}
	//判定
	int total = 0; //判定数
	int correct_n = 0; //正解数
	for (size_t i = 0; i < x_j.size(); ++i)
	{
		vector<double> output = nn.compute(x_j[i]); //NNで計算
													//NNの出力と一番近い回答点を見つける
													//NNの出力ベクトルと一番近い分類ベクトルを探す
		vector<double> dist_v(t_j[i].size()); //分類ベクトルとの距離のベクトル
		for (size_t j = 0; j < dist_v.size(); ++j){
			//分類ベクトル作成
			vector<double> t_ref(t_j[i].size());
			std::fill(t_ref.begin(), t_ref.end(), 0.0); //初期化
			t_ref[j] = 1.0; //classificationでは選択肢のj番目はj番目のみが1で他が0の教師ベクトルである

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
			++correct_n; //正解数を1増加
		else
		{   //不正解なら
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
}

