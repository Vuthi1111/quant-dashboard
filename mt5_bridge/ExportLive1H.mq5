//+------------------------------------------------------------------+
//|                                                ExportLive1H.mq5  |
//|                                  Machine Learning Regime Bridge  |
//+------------------------------------------------------------------+
#property copyright "Algo Trading Bridge"
#property link      ""
#property version   "1.00"
#property script_show_inputs

//--- input parameters
input string   OutFileName = "nas100_live.csv"; // File name to export
input int      BarsToExport = 1000;             // Number of 1H bars to export (roughly 40 days)

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   // We need the 1-Hour timeframe specifically
   ENUM_TIMEFRAMES tf = PERIOD_H1;
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   // Fetch the rates
   int copied = CopyRates(_Symbol, tf, 0, BarsToExport, rates);
   if(copied <= 0)
     {
      Print("Error copying rates: ", GetLastError());
      return;
     }
     
   // Open the file in the MQL5/Files/ directory
   int file_handle = FileOpen(OutFileName, FILE_WRITE | FILE_CSV | FILE_ANSI, '\t');
   if(file_handle == INVALID_HANDLE)
     {
      Print("Failed to open file for writing! Error: ", GetLastError());
      return;
     }
     
   // Write the exact header our Python pipeline expects
   FileWrite(file_handle, "datetime", "open", "high", "low", "close", "tick_volume", "spread", "real_volume");
   
   // Write data from oldest to newest (so Python reads it correctly as a time series)
   for(int i = copied - 1; i >= 0; i--)
     {
      string time_str = TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS);
      
      FileWrite(file_handle, 
                time_str, 
                DoubleToString(rates[i].open, _Digits),
                DoubleToString(rates[i].high, _Digits),
                DoubleToString(rates[i].low, _Digits),
                DoubleToString(rates[i].close, _Digits),
                IntegerToString(rates[i].tick_volume),
                IntegerToString(rates[i].spread),
                IntegerToString(rates[i].real_volume)
                );
     }
     
   FileClose(file_handle);
   Print("SUCCESS! Exported ", copied, " 1H bars to MQL5/Files/", OutFileName);
   Print("You can now run the Python Live Inference engine.");
  }
//+------------------------------------------------------------------+
