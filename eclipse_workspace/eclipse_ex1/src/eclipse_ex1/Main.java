package eclipse_ex1;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

	public static double[][] inputFile(String path) throws IOException{
		List<String[]> list = new ArrayList<String[]>();
		BufferedReader in = new BufferedReader(new FileReader(path));
		String line;
		while((line = in.readLine()) != null){
//			System.out.println(line);	//一行ずつprintできる
			list.add(line.split(" "));
		}
		in.close();
		double[][] x = new double[list.size()][2];
		for(int i=0; i<list.size(); i++){
			for (int j=0; j<list.get(i).length; j++){
				x[i][j] = Double.parseDouble(list.get(i)[j]);
			}
		}
		return x;
	}

	public static void main(String[] args) throws IOException {
		double[][] x;	//入力ベクトル

		//ファイル読み込み
		String path = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/data.dat";
		x = inputFile(path);

	}

}
