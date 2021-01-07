/* NN ���C�u�������g���₷���悤�ɕύX */
/* 7763c160b4ec1caa99718cd3c865339227a1908e */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>		// vector�p

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

/* ��O�����p */
void my_exception(const char *func, const std::string msg)
{
	std::stringstream sstr;
	sstr << "In [" << func << "], " << msg;
	throw std::runtime_error(sstr.str());
}

//�V�O���C�h�֐�
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/* �n���ꂽ vector �����t�@�C���ɏo�͂��� */
template <typename Seq>
void output_sequence(const Seq &seq, const std::string path)
{
	std::ofstream ofs(path);
	if (!ofs)
		my_exception(__FNAME__, "cannot create [" + path + "]");

	for (auto &&e : seq)
		ofs << e << std::endl;
}

/* �w�K�ς݂̃p�����[�^���t�@�C���ɏo�� */
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

/* str �� del �ŕ������� */
std::vector<double> NeuralNetwork::split(std::string str, char del)
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

//NeuralNetwork�̃R���X�g���N�^
//�d�݁A�t�`�d����덷�A�e�j���[�����̏o�͂̔z��𐶐�
NeuralNetwork::NeuralNetwork(const vector<size_t> &n_, double eta_, std::string nn_mode_)
	: n(n_), L(n.size()), eta(eta_), nn_mode(nn_mode_)
{
	init();
}

/* �w�K�ς݂̏d�݂��g�p���� */
NeuralNetwork::NeuralNetwork(const std::string nn_mode_, const std::string path_learned_param)
	: nn_mode(nn_mode_)
{
	std::ifstream ifs(path_learned_param);
	if (!ifs)
		my_exception(__FNAME__, "cannot open");
	std::string line;

	/* n */
	{
		std::getline(ifs, line);
		vector<double> splitted = split(line, ',');
		n.resize(splitted.size());
		for (size_t i = 0; i < splitted.size(); ++i)
			n[i] = (size_t)splitted[i];
	}

	/* L */
	L = n.size();

	/* eta */
	std::getline(ifs, line);
	eta = std::stod(line);

	/* �����o�ϐ��̏����� */
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

void NeuralNetwork::init()
{
	/* nn_mode �̃G���[�`�F�b�N */
	if (nn_mode != MODE_C and nn_mode != MODE_R)
		my_exception(__FNAME__, "invalid nn_mode");

	z.resize(L);
	for (size_t l = 0; l < L; ++l)
	{
		z[l].resize(n[l] + 1); //�o�C�A�X���o�͂���j���[������]���Ɋm��
		z[l][0] = 1.0;		   //�o�C�A�X�j���[�����̏o�͂�ݒ�
	}
	d.resize(L);
	for (size_t l = 1; l < L; ++l)
	{						   //d[0][i]�͓��͑w������덷�v�Z�͕K�v�Ȃ�
		d[l].resize(n[l] + 1); //�t�`�d����덷���e�w�̊e�j���[�����ɗp��	�Y�����ɒ���
		d[l][0] = 0.0;		   //�e�w��0�Ԗڂ̃j���[��������͌덷���`�d���Ȃ�
	}
	w.resize(L - 1); //L�w���邩��w�Ԃ̐���L-1
	for (size_t l = 0; l < L - 1; ++l)
	{
		w[l].resize(n[l + 1] + 1); //��l�w�����l+1�w�ւ̏d�݂��m�ہ@����Ƀo�C�A�X�j���[��������̏d�݂��]���Ɋm��
		for (size_t k = 0; k < n[l + 1] + 1; ++k)
		{
			w[l][k].resize(n[l] + 1); //�Y�����ɒ��� �^�񒆂̓Y��������l+1�w�̃j���[�����ōŌ�̓Y��������l�w�̃j���[����
			for (size_t i = 0; i < n[l] + 1; ++i)
			{
				w[l][0][i] = 0.0; //l+1�w�ڂ̃o�C�A�X�j���[�����ւ͏o�͂��Ȃ��@w[l]�͂��ꂼ��s��ɂȂ�
			}
		}
	}
	reset();
}

void NeuralNetwork::reset()
{
	//�����Z���k�c�C�X�^�𗐐�������ŏ�����
	//-1.0�`1.0�̈�l���z���g�p
	std::random_device rnd;
	std::array<std::random_device::result_type, std::mt19937::state_size> v;
	std::generate(v.begin(), v.end(), std::ref(rnd));
	std::seed_seq seed(v.begin(), v.end());
	std::mt19937 engine(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	for (size_t l = 0; l < L - 1; ++l)
		for (size_t k = 1; k < n[l + 1] + 1; ++k)
			for (size_t i = 0; i < n[l] + 1; ++i)
				w[l][k][i] = dist(engine); //������
}

std::vector<double> NeuralNetwork::compute(const vector<double> &x)
{
	if (x.size() != n[0])
	{
		std::stringstream sstr;
		sstr << "���̓f�[�^�̎������s�K�؂ł��D\n"
			 << "�Z�b�g���ꂽ�p�����[�^�̎����F" << n[0]
			 << ", ���͂��ꂽ�f�[�^�̎����F" << x.size();
		my_exception(__FNAME__, sstr.str());
	}

	for (size_t i = 1; i < n[0] + 1; ++i)
	{
		z[0][i] = x[i - 1]; //��0�w�֓��̓x�N�g�����Z�b�g
	}
	for (size_t l = 1; l < L; ++l)
	{ //�������v�Z ��l�w��k�Ԗڂ̃j���[�����̏o�͂��v�Z
		for (size_t k = 1; k < n[l] + 1; ++k)
		{
			double sum = 0.0;
			for (size_t i = 0; i < n[l - 1] + 1; ++i)
			{
				sum += z[l - 1][i] * w[l - 1][k][i];
			}

			/* �o�͂̌v�Z */
			if (nn_mode == MODE_R and l == L - 1)
				z[l][k] = sum; // �o�͑w�����P���ʑ�
			else
				z[l][k] = sigmoid(sum);
		}
	}
	vector<double> output(z[L - 1].begin() + 1, z[L - 1].end()); //�ŏI�w�̏o�͂��i�[
	return std::move(output);
}

void NeuralNetwork::back_propagation(const vector<double> &t)
{
	//�o�͑w�ɂ��Čv�Z�A�d�݂��X�V
	for (size_t j = 1; j < n[L - 1] + 1; ++j)
	{
		if (nn_mode == MODE_C)
			d[L - 1][j] = -(t[j - 1] - z[L - 1][j]) * z[L - 1][j] * (1.0 - z[L - 1][j]);
		else
			d[L - 1][j] = -(t[j - 1] - z[L - 1][j]); /* ��A���͂ł͏o�͑w�̊������֐��Ƃ��čP���֐���p���邽�� */
		for (size_t i = 0; i < n[L - 2] + 1; ++i)
		{
			w[L - 2][j][i] -= eta * d[L - 1][j] * z[L - 2][i];
		}
	}
	//���͑w����o�͑w�̎�O�܂ł̏d�݂��X�V
	for (size_t l = L - 2; l != 0; --l)
	{ //��0�w�̌덷�͌v�Z���Ȃ�
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

void NeuralNetwork::autoencoder(const vector<vector<double>> &x_v, int l, double epsilon, int limit, const std::string convergence_mode)
{
	/* nn_mode �̃G���[�`�F�b�N */
	if (nn_mode == MODE_R)
		my_exception(__FNAME__, "��A���[�h�ł̓���͖��m�F�ł�");

	//���ȕ���������쐬����
	//��l-1�w�Ƒ�l�w���P������
	//l�w����l_decode�w�͍P���ʑ��Ƃ���
	//decoder�̂��߂̃j���[�����o�͂ƃf���^�̃x�N�g�����m��
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
			//���̓x�N�g�����Z�b�g
			for (size_t i = 1; i < n[l - 1] + 1; ++i)
			{
				z[l - 1][i] = x_v[p][i - 1];
			}
			//�܂����ȕ��������encode���v�Z����
			for (size_t i = 1; i < n[l] + 1; ++i)
			{
				double sum = 0.0;
				for (size_t h = 0; h < n[l - 1] + 1; ++h)
				{
					sum += z[l - 1][h] * w[l - 1][i][h];
				}
				z[l][i] = sigmoid(sum);
			}
			//���Ɏ��ȕ��������decode���v�Z����
			for (size_t j = 1; j < n[l - 1] + 1; ++j)
			{
				z_ae[j] = 0.0;
				for (size_t i = 1; i < n[l] + 1; ++i)
				{
					z_ae[j] += z[l][i] * w[l - 1][i][j];
				}
			}
			//�e���͂ɑ΂��Č덷�t�`�d�@�ŏd�݂��X�V����
			//�o�͑w�ł̃f���^d_ae�͎��̂悤�ɂȂ�
			for (size_t j = 1; j < n[l - 1] + 1; ++j)
			{
				d_ae[j] = z_ae[j] - z[l - 1][j];
			}
			//�d�ݍX�V�@�d�݋��L�����邱�ƂōX�V����̂�decoder�w�����ł悢
			for (size_t j = 1; j < n[l - 1] + 1; ++j)
			{
				for (size_t i = 0; i < n[l] + 1; ++i)
				{
					w[l - 1][i][j] -= eta * d_ae[j] * z[l][i];
				}
				//�덷�֐����o��
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

std::vector<double> NeuralNetwork::compute_lth_layer_output(const vector<double> &x, int u)
{
	/* nn_mode �̃G���[�`�F�b�N */
	if (nn_mode == MODE_R)
		my_exception(__FNAME__, "��A���[�h�ł̓���͖��m�F�ł�");

	for (size_t i = 1; i < n[0] + 1; ++i)
	{
		z[0][i] = x[i - 1]; //��0�w�֓��̓x�N�g�����Z�b�g
	}

	for (int l = 1; l < u + 1; ++l)
	{ //�������v�Z ��l�w��k�Ԗڂ̃j���[�����̏o�͂��v�Z
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
	vector<double> output(z[u].begin() + 1, z[u].end()); //�ŏI�w�̏o�͂��i�[
	return std::move(output);
}

void NeuralNetwork::prelearning(const vector<vector<double>> &x_v, double epsilon, int limit, const std::string convergence_mode)
{
	/* nn_mode �̃G���[�`�F�b�N */
	if (nn_mode == MODE_R)
		my_exception(__FNAME__, "��A���[�h�ł̎��O�w�K�̓���͖��m�F�ł�");

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

double NeuralNetwork::error(const vector<double> &t) const
{
	double sum = 0.0;
	for (size_t j = 1; j < n[L - 1] + 1; ++j)
		sum += std::pow((t[j - 1] - z[L - 1][j]), 2); //���t�x�N�g���̓Y������0����n�܂�B����o�͂�0�Ԗڂ̓o�C�A�X���ƂȂ��Ă���
	return 0.5 * sum;
}

double NeuralNetwork::learning(const vector<vector<double>> &x_v,
							   const vector<vector<double>> &t_v,
							   const std::string path_E_his,
							   const std::string path_learned_param,
							   const std::string convergence_mode,
							   double epsilon,
							   int limit)
{

	vector<double> E_his;
	int count = 0; // ���s�񐔃J�E���^
	while (true)
	{
		double E = 0.0;
		for (size_t p = 0; p < x_v.size(); ++p)
		{							  //p�Ԗڂ̓��̓x�N�g���Ɠ��͑w�̃j���[���������r���������G���[
			compute(x_v[p]);		  //�������v�Z
			back_propagation(t_v[p]); //�t�덷�`�d�v�Z
			E += error(t_v[p]);		  //�G���[���v�Z
		}
		E_his.push_back(E); //�G���[�����ɕۑ�
		++count;

		/* �������� */
		if (convergence_mode == "deltaE") //�G���[�̕ω����Ŕ��肷����@
		{
			if (count != 1)
			{
				double deltaE = std::abs(E_his[E_his.size() - 1] - E_his[E_his.size() - 2]);
				if (deltaE < epsilon)
					break;
			}
		}
		else if (E < epsilon) //�G���[�̑傫���Ŕ��肷����@
			break;
		if (count == limit) //���s�񐔂őł��؂�
			break;
	}

	/* �G���[�������t�@�C���ɏo�� */
	if (path_E_his != "")
		output_sequence(E_his, path_E_his);

	/* �w�K�ς݂̃p�����[�^���t�@�C���ɏo�� */
	if (path_learned_param != "")
		output_learned_param(n, eta, w, path_learned_param);

	return E_his.back();
}

// int main(void) {}

namespace py = pybind11;
using namespace py::literals; // py::arg ���ȗ����邽��
PYBIND11_MODULE(so, m)
{
	m.doc() = "so::NeuralNetwork";
	py::class_<NeuralNetwork>(m, "NeuralNetwork")
		.def(py::init<const std::vector<size_t> &, double, std::string>())
		.def(py::init<const std::string, const std::string>())
		.def("learning", &NeuralNetwork::learning,
			 "x_v"_a,
			 "t_v"_a,
			 "path_E_his"_a = "",
			 "path_learned_param"_a = "",
			 "convergence_mode"_a = "deltaE",
			 "epsilon"_a = 1E-4,
			 "limit"_a = -1)
		.def("prelearning", &NeuralNetwork::prelearning,
			 "x_v"_a,
			 "epsilon"_a,
			 "limit"_a = -1,
			 "convergence_mode"_a = "deltaE")
		.def("compute", &NeuralNetwork::compute)
		.def("reset", &NeuralNetwork::reset);
}