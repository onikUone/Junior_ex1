package neuralNetwork;

public class NeuralNetwork {
	//paramater
	int n_input;
	int n_hidden;
	int n_output;
	double eta;
	double alpha;
	int count_train;

	InputNeuron[] input;
	HiddenNeuron[] hidden;
	OutputNeuron[] output;

	double[][] x;
	double[][] y;

	MersenneTwisterFast mtf;

	//method
	public void setCount(int _count_train) {
		this.count_train = _count_train;
	}

	public void setSeed(int _seed) {
		mtf.setSeed(_seed);
	}

	public void setTeacher(double[][] _x, double[][] _y) {
		this.x = _x;
		this.y = _y;
	}

	public void createNetwork() {
		input = new InputNeuron[n_input];
		hidden = new HiddenNeuron[n_hidden];
		output = new OutputNeuron[n_output];

		for (int i = 0; i < n_input; i++) {
			input[i] = new InputNeuron();
		}
		for (int i = 0; i < n_hidden; i++) {
			hidden[i] = new HiddenNeuron(n_input, mtf.nextDoubleIE(), mtf.nextDoubleIE(), eta, alpha);
		}
		for (int i = 0; i < n_output; i++) {
			output[i] = new OutputNeuron(n_hidden, mtf.nextDoubleIE(), mtf.nextDoubleIE(), eta, alpha);
		}
	}

	public void forward() {
		for (int i = 0; i < n_hidden; i++) {
			hidden[i].forward_function(input);
		}
		for (int i = 0; i < n_output; i++) {
			output[i].forward_function(hidden);
		}
	}

	public void train() {
		for (int i = 0; i < count_train; i++) {
			for(int j=0; j< x.length; j++) {
				//入力層入力
				for(int k = 0; k<n_input; k++) {
					input[k].input(x[j][k]);
				}
				//順方向計算
				forward();
				//BackPropagation
				//中間層重み更新
				for(int k= 0; k<n_hidden; k++) {
					hidden[k].reWeight(input, output, y[j], k);
				}
				//出力層重み更新
				for(int k=0; k<n_output; k++) {
					output[k].reWeight(hidden, y[j][k]);
				}
			}
		}
	}

	public double evaluation() {
		double e = 0.0;
		for(int i=0; i<x.length; i++) {
			for(int j= 0; j< n_input; j++) {
				input[j].input(x[i][j]);
			}
			forward();
			for(int j= 0; j<n_output; j++) {
				e += (y[i][j] - output[j].output()) * (y[i][j] - output[j].output()) / 2;
			}
		}
		return e;
	}

	//constractor
	NeuralNetwork(int _n_input, int _n_hidden, int _n_output, double _eta, double _alpha) {
		this.n_input = _n_input;
		this.n_hidden = _n_hidden;
		this.n_output = _n_output;
		this.eta = _eta;
		this.alpha = _alpha;

		mtf = new MersenneTwisterFast();
	}
}
