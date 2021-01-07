#ifdef __GNUC__ /* GCC */
#define __FNAME__ __PRETTY_FUNCTION__
#elif _MSC_VER /* Visual C++ */
#define __FNAME__ __FUNCSIG__
#else
#define __FNAME__ __func__
#endif
#include <vector>
#include <string>
#include <ostream>

namespace so
{
	std::vector<double> split(std::string str, char del);

	class NeuralNetwork
	{
	private:
		template <typename T>
		using vector = std::vector<T>;
		using size_t = std::size_t;

	public:
		/*-----------------------------------------------
		*
		* コンストラクタ
		* n_：ニューロン数
		* eta：学習係数
		* nn_mode：classification もしくは regression を指定
		*
		-----------------------------------------------*/
		explicit NeuralNetwork(const vector<size_t> &n_, double eta_, std::string nn_mode_);
		explicit NeuralNetwork(const std::string nn_mode_, const std::string path_learned_param);

		//学習関数
		double learning(const vector<vector<double>> &x_v,				/* 入力ベクトル */
						const vector<vector<double>> &t_v,				/* 教師ベクトル */
						const std::string path_E_his = "",				/* 全データの誤差 E の履歴出力先 */
						const std::string path_learned_param = "",		/*  学習済みのパラメータの出力先 */
						const std::string convergence_mode = "deltaE", /* 収束条件（ここを変えると収束判定の方法が変わる） */
						double epsilon = 1E-4,							/* 収束判定値 */
						int limit = -1);								/* 打ち切り試行回数 */

		//自己符号化による事前学習
		void prelearning(const vector<vector<double>> &x_v,
						 double epsilon,
						 int limit = -1,
						 const std::string convergence_mode = "deltaE");

		//順方向計算
		vector<double> compute(const vector<double> &x); //入力ベクトル
		void reset();

	private:
		/* メンバ変数の初期化 */
		void init();

		//自己符号化器
		void autoencoder(const vector<vector<double>> &x_v, int l, double epsilon, int limit, const std::string convergence_mode); //引数は入力ベクトルと何層目で自己符号化器を作るか　番号は1からL-1まで

		//第l層までの順方向計算
		vector<double> compute_lth_layer_output(const vector<double> &x, int l);

		//ニューロン一層に誤差逆伝搬法で重みを更新する関数
		void back_propagation(const vector<double> &t); //引数は教師ベクトル

		//誤差e算出
		double error(const vector<double> &t) const; //引数は教師ベクトル

		//データメンバ
		vector<vector<double>> z;		  //ニューロン出力
		vector<vector<double>> d;		  //誤差逆伝播法で逆伝播する誤差
		vector<vector<vector<double>>> w; //重み係数
		vector<size_t> n;				  //ニューロン数のベクトル
		size_t L;						  //層数
		double eta;						  //学習係数

		const std::string nn_mode;					 /* NN のモード（分類か回帰か） */
		const std::string MODE_C = "classification"; /* 分類モード */
		const std::string MODE_R = "regression";	 /* 回帰モード */
	};
} //namespace so
