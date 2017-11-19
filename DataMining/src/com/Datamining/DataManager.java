package com.Datamining;


import java.util.*;
import java.io.*;
import java.nio.file.*;

import org.jblas.*;

public final class DataManager {

	//fail?
	public static List<HashMap<String,Integer>> count = new ArrayList<HashMap<String,Integer>>();
	
	private static String regex = "";
	
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
	
	public static void SetRegex(String reg) {
		regex = reg;
	}	
	
	//read csv
	public static ArrayList<String[]> ReadCSV(String dir,boolean skip) {
		count.clear();
		
		ArrayList<String[]> dataSet = new  ArrayList<String[]>();
		Path path = Paths.get(dir);
		
		try (BufferedReader reader = Files.newBufferedReader(path)){
			String line = null;
			
			String[] attributes = (reader.readLine()).split(",");
			
			for(String a : attributes) {
				count.add(new HashMap<String,Integer>());
			}
			
			while ((line = reader.readLine()) != null) {
				
				//will read next line if skip = true and line has regex value
				if(!(line.contains(regex) && skip)) {
					
					String[] value = line.split(",");
					dataSet.add(value);
					
					for(int i = 0; i < value.length; i++) {
						if(!value[i].equals(regex)) {
							count.get(i).put(value[i], count.get(i).containsKey(value[i]) ? count.get(i).get(value[i]) + 1 : 1);
						}
					}
					
				}
			}
			
		} catch (IOException e) {
			System.out.println("Bad Directory. Cannot find >>> '" + dir+ "'");
		}
		
		
		int col =0;
		for(int a= 0;a < count.size(); a++) {
			for(Map.Entry<String, Integer> set : count.get(a).entrySet()) {
				if(set.getKey().equals(regex)) {
					System.out.println(set.getKey() + " : " + set.getValue() + " : " + col + " : " + a);
				}
				col ++;
			}
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
	
	//!?!
	private static Map.Entry<String,Integer> getCommon(HashMap<String,Integer> col) {
		
		Map.Entry<String, Integer> maxEntry = null;

		for (Map.Entry<String, Integer> entry : col.entrySet()) {
		    if (maxEntry == null || entry.getValue() > maxEntry.getValue()) {
		        maxEntry = entry;
		    }
		}
		
		return maxEntry;
	}
	
	
	//get 'stratified' sampling : fake, roll 0 to 1 and added in if lower than sampleRatio
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
	//
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
	
	
	//%$£%$£"F^
	//convert string attributes into double
	//no cell can be Null for this
	//I dont know how to evaluate String values, therefore they are assigned with integers depending on their value.
	//requires to know which line is classtype. Nope.
	public static DoubleMatrix dataSetToMatrix(ArrayList<String[]> dataSet , int classType) {
		
		HashMap<String, Double> StringToNumber = new HashMap<String, Double>();
		
		//for(HashMap<String,Integer> col : count) {
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
	public static List<DoubleMatrix> splitFold(DoubleMatrix dataSet,int folds) { 
	
		List<DoubleMatrix> retunrBins = new ArrayList<DoubleMatrix>();
		
		
		
		for(int i = 0; i < folds; i++) {
			retunrBins.add(new DoubleMatrix(0, dataSet.columns));
		}
		
		for(int i = 0; i < dataSet.rows; i++) {
			int bin = i % folds;
			retunrBins.set(bin, DoubleMatrix.concatVertically(retunrBins.get(bin), dataSet.getRow(i)));
		}
		
		
		
		return retunrBins;
	}
	
	
	public static void saveRecord(String dir,int KNN,boolean weighted, int[][] confusionMat,List<String> classType, double accuracy) {
		try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(dir, false)))) {
			writer.println("TestSet");
			writer.println("");
			writer.println("TestSet:");
			writer.println("accuracy: " + accuracy);
			writer.println("ConfusionMatrix:");
			
			String CM = "True: ";
			for(String clstyp : classType) {
				CM = CM + " " + clstyp;
			}
			writer.println(CM);
			
			for(int i = 0; i < confusionMat.length; i++) {
				CM = classType.get(i) + ": ";
				for (int j = 0; j < confusionMat[i].length; j++) {
					CM = CM + " " +  confusionMat[i][j];
				}
				writer.println(CM);
			}
			
			writer.println("Best K: " + KNN);
			writer.println("Weighted: " + weighted);
			
			writer.close();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	
	
	
	
}
