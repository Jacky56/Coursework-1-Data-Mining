package com.Datamining;


import java.util.*;
import java.io.*;
import java.nio.file.*;

import org.jblas.*;



//main function: to manage data.
public final class DataManager {

	//counts unique values and how many times they appeared in dataset
	public static List<LinkedHashMap<String,Integer>> count = new ArrayList<LinkedHashMap<String,Integer>>();
	
	
	private static String regex = "";
	
	//used for splitfold to get deterministic value 
	private static long seed = 0l;
	
	//encapsulation 
	public static DoubleMatrix GetFeatures(DoubleMatrix dataSet,int[] col) {
		DoubleMatrix X = new DoubleMatrix(dataSet.rows,0);
		for(int i = 0; i < dataSet.columns; i++) {
			boolean hasException = false;
			
			for(int exception : col) {
				if(i == exception) {
					hasException = true;
				}
			}
			if(!hasException) {
				X = DoubleMatrix.concatHorizontally(X, dataSet.getColumn(i));
			}
			
		}
		return X;
	}
	
	//encapsulation
	public static DoubleMatrix GetClass(DoubleMatrix dataSet,int col) {
		return dataSet.getColumn(col);
	}
	
	//encapsulation
	public static void SetRegex(String reg) {
		regex = reg;
	}	
	
	//encapsulation
	public static void SetSeed(long seeding) {
		seed = seeding;
	}	
	
	
	//read csv
	public static ArrayList<String[]> ReadCSV(String dir,boolean skip) {
		
		ArrayList<String[]> dataSet = new  ArrayList<String[]>();
		Path path = Paths.get(dir);
		
		try (BufferedReader reader = Files.newBufferedReader(path)){
			String line = null;
			
			String[] attributes = (reader.readLine()).split(",");
			
			for(String a : attributes) {
				count.add(new LinkedHashMap<String,Integer>());
			}
			
			while ((line = reader.readLine()) != null) {
				
				if (line.matches(".*\\w.*")) {
					//will read next line if skip = true and line has regex value
					if(!(line.contains(regex) && skip)) {
						
						String[] value = line.split(",");
						
						//cleanse string
						for(int i = 0; i < value.length; i++) {
							value[i] = value[i].trim().toLowerCase();
						}
						
						dataSet.add(value);
						for(int i = 0; i < value.length; i++) {
							if(!value[i].equals(regex)) {
								count.get(i).put(value[i], count.get(i).containsKey(value[i]) ? count.get(i).get(value[i]) + 1 : 1);
							}
						}
						
					}
				} 
			}
			
		} catch (IOException e) {
			System.out.println("Bad Directory. Cannot find >>> '" + dir+ "'");
		}
		
		return dataSet;
	}
	
	
	
	
	//replace regex/Null value with common/mode value in column
	public static ArrayList<String[]> ReplaceMissingValues(ArrayList<String[]> dataSet) {
		ArrayList<String[]> returnSet = dataSet;
		
		for(int row = 0; row < returnSet.size(); row++) {
			for(int col = 0; col < returnSet.get(row).length; col ++) {
				
				if(returnSet.get(row)[col].equals(regex)) {
					returnSet.get(row)[col] = getCommon(count.get(col)).getKey();
				}
				
			}
		}
		return returnSet;
	}
	
	//get most common using 'count'
	private static Map.Entry<String,Integer> getCommon(HashMap<String,Integer> col) {
		
		Map.Entry<String, Integer> maxEntry = null;

		for (Map.Entry<String, Integer> entry : col.entrySet()) {
		    if (maxEntry == null || entry.getValue() > maxEntry.getValue()) {
		        maxEntry = entry;
		    }
		}
		
		return maxEntry;
	}
	
	//both of you are gonna write my report for ML and AI :)))))))))))))))))))))))))))))
	
	
	//get 'stratified' sampling : fake, roll 0 to 1 and added in if lower than sampleRatio
	//do not use.
	public static ArrayList<String[]> Sample(ArrayList<String[]> dataSet, double sampleRatio) {
		ArrayList<String[]> returnSet = new ArrayList<String[]>();
		
		for(String[] row : dataSet) {
			if(Math.random() <= sampleRatio) {
				returnSet.add(row);
			}
		}
		return returnSet;
	}
	
	
	
	//no cell can be Null for this function
	//Z-normalization Z = (X-u) /s
	public static DoubleMatrix Normalize(DoubleMatrix dataSet) {
		
		DoubleMatrix returnSet = new DoubleMatrix(dataSet.rows,dataSet.columns);
		DoubleMatrix rawSet = dataSet;
		
//		for(int row = 0; row < dataSet.rows; row++) {
//			DoubleMatrix X = rawSet.getRow(row);
//			double u = rawSet.getRow(row).mean();
//			double s = GetStd(rawSet.getRow(row));
//			DoubleMatrix Z = (X.sub(u)).div(s);
//			returnSet.putRow(row, Z);
//		}
		
		for(int col = 0; col < dataSet.columns; col ++) {
			
			DoubleMatrix X = rawSet.getColumn(col);
			double u = X.mean();
			double s = GetStd(X);
			DoubleMatrix Z = (X.sub(u)).div(s);
			returnSet.putColumn(col, Z);
		}
		
		return returnSet;
	}
	
	
	public static double GetStd(DoubleMatrix s) {
		double u = s.mean();
		double v = 0;
		for(double i : s.toArray()) {
			v += Math.pow(i - u, 2);
		}
		return Math.sqrt(v/(s.toArray().length -1));
	}
	
	
	//convert string attributes into double
	//no cell can be Null for this
	//I dont know how to evaluate String values, therefore they are assigned with integers depending on their value.
	//requires to know which line is classtype.
	public static DoubleMatrix dataSetToMatrix(ArrayList<String[]> dataSet , int classType) {
		
		LinkedHashMap<String, Double> StringToNumber = new LinkedHashMap<String, Double>();
		
		for(int col = 0; col < count.size(); col ++) {	

			Double val = 0d;
			for (Map.Entry<String, Integer> entry : count.get(col).entrySet()) {
				try {
					Double.valueOf(entry.getKey());
				} catch (NumberFormatException e) {
					if(!StringToNumber.containsKey(entry.getKey())) {
						StringToNumber.put(entry.getKey(), val);
						if(col == classType) {
							val ++;
						} else {
							
							//String to number here !!!!! <--------
							val ++;
							//String to number here !!!!! <--------
							
						}
					}
				}
			}
		}
		
		DoubleMatrix returnSet = new DoubleMatrix(dataSet.size(),dataSet.get(0).length);
		
		for(int row = 0; row < dataSet.size(); row++) {
			ArrayList<Double> conversion = new ArrayList<Double>();
			for(String col : dataSet.get(row)) {
				try {
					conversion.add(Double.valueOf(col));
				} catch(NumberFormatException e) {
					conversion.add(StringToNumber.get(col));
				} catch (Exception e) {
					System.out.println("Null values");
				}
			}
			
			if(conversion.size() < dataSet.get(0).length) {
				System.out.println(conversion + "   " + row + "   " + dataSet.size());
			}
			
			returnSet.putRow(row,new DoubleMatrix(conversion));
		}
		
		return returnSet;
	}
	
	
	//split data set to sample set depending on column picked
	public static List<DoubleMatrix> split(DoubleMatrix dataSet,int col) {
		
		HashMap<Double ,DoubleMatrix> bins = new HashMap<Double ,DoubleMatrix>();
		

		for (Map.Entry<String, Integer> entry : count.get(col).entrySet()) {
			bins.put(Double.valueOf(entry.getKey()), new DoubleMatrix(0, dataSet.columns));
		}		
		
		for(int row = 0; row < dataSet.rows; row++) {
			double binNo = dataSet.get(row,col);
			bins.put(binNo, DoubleMatrix.concatVertically(bins.get(binNo), dataSet.getRow(row)));
		}
		
		
		List<DoubleMatrix> returnBins = new ArrayList<DoubleMatrix>();
		for (DoubleMatrix entry : bins.values()) {
			returnBins.add(entry);
		}
		return returnBins;
	}
	
	
	//do not use for this task.
	//split dataset amongst several folds with random permutation
	public static List<DoubleMatrix> splitFold(DoubleMatrix dataSet,int folds, boolean shuffle) { 
	
		DoubleMatrix set = dataSet;
		List<DoubleMatrix> retunrBins = new ArrayList<DoubleMatrix>();
		
		if(shuffle) {
			List<Integer> permutation = new ArrayList<Integer>();
			for(int i = 0; i < set.rows; i++) {
				permutation.add(i);
			}
			
			Collections.shuffle(permutation, new Random(seed));
			
			set = new DoubleMatrix(dataSet.rows,dataSet.columns);
			for(int i = 0; i < set.rows; i++) {
				set.putRow(i, dataSet.getRow(permutation.get(i)));
			}
		}
		
		for(int i = 0; i < folds; i++) {
			retunrBins.add(new DoubleMatrix(0, set.columns));
		}
		for(int i = 0; i < set.rows; i++) {
			int bin = i % folds;
			retunrBins.set(bin, DoubleMatrix.concatVertically(retunrBins.get(bin), set.getRow(i)));
		}
		
		return retunrBins;
	}
	
	

	
	
	public static int[][] combineMat(List<Record[]> records) {
		int[][] mat = new int[records.get(0)[0].confusionMat.length][records.get(0)[0].confusionMat.length];
		
		for(Record[] a : records) {
			for(Record b : a) {
				for(int i = 0; i < 2; i++) {
					for(int j = 0; j < 2; j++) {
						mat[i][j] += b.confusionMat[i][j];
					}
				}
			}
		}
		return mat;
	}
	
	
	public static double getAccuracy(int[][] mat) {
		int correct = 0;
		int total = 0;
		for(int i = 0; i < 2; i++) {
			for(int j = 0; j < 2; j++) {
				total += mat[i][j];
				if(i == j) {
					correct += mat[i][j];
				}
			}
		}
		
		return (double)correct / total;
	}
	
	
	public static double GetStd(List<Record[]> records) {
		double sum = 0;
		int item = 0;
		
		for(Record[] a : records) {
			for(Record b : a) {
				sum += b.accuracy;
				item ++;
			}
		}
		
		double u = sum / item;
		double v = 0;
		for(Record[] a : records) {
			for(Record b : a) {
				v += Math.pow(b.accuracy - u, 2);
			}
		}
		return Math.sqrt(v/(item - 1));
	}
	
	
	public static void saveRecord(String dir,String positiveClass, int KNN,boolean weighted, List<Record[]> validation, List<Record[]> test,List<String> classType, int fold, double time, int threadPool) {
		try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(dir, false)))) {
			
			String cleanString = positiveClass.trim().toLowerCase();
			int nC = -1;
			int pC = -1;
			for(int i = 0; i < classType.size(); i++) {
				if(classType.get(i).equals(cleanString)) {
					pC = i;
				} else {
					nC = i;
				}
			}
			
			int[][] validationM = combineMat(validation);
			int[][] testM = combineMat(test);
			double validationstd = GetStd(validation);	
			
			
			
			writer.println("ValidationSet");
			writer.println("");
			writer.println("seed: " + seed);
			writer.println("folds: " + fold);
			writer.println("accuracy: " + getAccuracy(validationM) + " +/- " + validationstd);
			writer.println("ConfusionMatrix:");
			String CM = "True: ";
			for(String clstyp : classType) {
				CM = CM + " " + clstyp;
			}
			writer.println(CM);
			
			for(int i = 0; i < validationM.length; i++) {
				CM = classType.get(i) + ": ";
				for (int j = 0; j < validationM[i].length; j++) {
					CM = CM + " " +  validationM[i][j];
				}
				writer.println(CM);
			}	
			writer.println("Best K: " + KNN);
			writer.println("Weighted: " + weighted);
			
			
			
			writer.println("");
			writer.println("TestSet");
			writer.println("");
			writer.println("TestSet:");
			writer.println("accuracy: " + getAccuracy(testM));
			writer.println("ConfusionMatrix:");
			
			CM = "True: ";
			for(String clstyp : classType) {
				CM = CM + " " + clstyp;
			}
			writer.println(CM);
			
			for(int i = 0; i < testM.length; i++) {
				CM = classType.get(i) + ": ";
				for (int j = 0; j < testM[i].length; j++) {
					CM = CM + " " +  testM[i][j];
				}
				writer.println(CM);
			}
			
			
			double total = 0;
			for(int i = 0; i < testM.length; i++) {
				total += testM[i][pC];
			}
			double precision = testM[pC][pC] / total;
			
			total = 0;
			for(int i = 0; i < testM.length; i++) {
				total += testM[nC][i];
			}
			double specificity = testM[nC][nC] / total;	
			
			total = 0;
			for(int i = 0; i < testM.length; i++) {
				total += testM[pC][i];
			}
			double sensitivity = testM[pC][pC] / total;				
			
			writer.println("<=50k performance measures");
			writer.println("precision:" + precision);
			writer.println("sensitivity:" + sensitivity);
			writer.println("specificity:" + specificity);
			
			
			writer.println("");
			writer.println("Multithreading");
			writer.println("");
			writer.println("Run time(millisecond): " + time);
			writer.println("Thread pool: " + threadPool);
			writer.println("Cores: " + Runtime.getRuntime().availableProcessors());
			
			writer.close();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	
	
	
	
}
