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

class NeuralNetwork
{
private:
	template <typename T>
	using vector = std::vector<T>;
	using size_t = std::size_t;

public:
	/*-----------------------------------------------
	*
	* �R���X�g���N�^
	* n_�F�j���[������
	* eta�F�w�K�W��
	* nn_mode�Fclassification �������� regression ���w��
	*
	-----------------------------------------------*/
	explicit NeuralNetwork(const vector<size_t> &n_, double eta_, std::string nn_mode_);
	explicit NeuralNetwork(const std::string nn_mode_, const std::string path_learned_param);

	//�w�K�֐�
	double learning(const vector<vector<double>> &x_v,				/* ���̓x�N�g�� */
					const vector<vector<double>> &t_v,				/* ���t�x�N�g�� */
					const std::string path_E_his = "",				/* �S�f�[�^�̌덷 E �̗����o�͐� */
					const std::string path_learned_param = "",		/*  �w�K�ς݂̃p�����[�^�̏o�͐� */
					const std::string convergence_mode = "deltaE", /* ���������i������ς���Ǝ�������̕��@���ς��j */
					double epsilon = 1E-4,							/* ��������l */
					int limit = -1);								/* �ł��؂莎�s�� */

	//���ȕ������ɂ�鎖�O�w�K
	void prelearning(const vector<vector<double>> &x_v,
					 double epsilon,
					 int limit = -1,
					 const std::string convergence_mode = "deltaE");

	//�������v�Z
	vector<double> compute(const vector<double> &x); //���̓x�N�g��
	void reset();

private:
	/* �����o�ϐ��̏����� */
	void init();

	//���ȕ�������C�����͓��̓x�N�g���Ɖ��w�ڂŎ��ȕ����������邩�C�ԍ���1����L-1�܂�
	void autoencoder(const vector<vector<double>> &x_v,
					 int l,
					 double epsilon,
					 int limit,
					 const std::string convergence_mode);

	//��l�w�܂ł̏������v�Z
	vector<double> compute_lth_layer_output(const vector<double> &x, int l);

	//�j���[������w�Ɍ덷�t�`���@�ŏd�݂��X�V����֐��C�����͋��t�x�N�g��
	void back_propagation(const vector<double> &t);

	//�덷e�Z�o�C�����͋��t�x�N�g��
	double error(const vector<double> &t) const;

	/* �����̕����� str �� del �ŋ�؂� */
	std::vector<double> split(std::string str, char del);

	//�f�[�^�����o
	vector<vector<double>>
		z;							  //�j���[�����o��
	vector<vector<double>> d;		  //�덷�t�`�d�@�ŋt�`�d����덷
	vector<vector<vector<double>>> w; //�d�݌W��
	vector<size_t> n;				  //�j���[�������̃x�N�g��
	size_t L;						  //�w��
	double eta;						  //�w�K�W��

	const std::string nn_mode;					 /* NN �̃��[�h�i���ނ���A���j */
	const std::string MODE_C = "classification"; /* ���ރ��[�h */
	const std::string MODE_R = "regression";	 /* ��A���[�h */
};