//+------------------------------------------------------------------+
//|                                                 ExportLiveEA.mq5 |
//|                                           Volatility Regime AI   |
//+------------------------------------------------------------------+
#property copyright "Volatility Regime AI Dashboard"
#property link      "https://github.com/Antigravity"
#property version   "1.00"

// --- Input Parameters ---
input int    InpBarsToExport = 10000;              // Number of bars to export (Historical Buffer)
input string InpFileName     = "nas100_live.csv";  // Output file name

// --- Global Variables ---
datetime last_write_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("ExportLiveEA Initialized.");
   Print("Will export ", InpBarsToExport, " bars to ", InpFileName, " on every tick (max 1 write/sec).");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("ExportLiveEA Removed.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime current_time = TimeCurrent();
   
   // --- Safety Throttle ---
   // MT5 can receive up to 50 ticks per second. 
   // Writing a 1MB file 50 times a second will freeze your Mac.
   // This ensures we only write to the hard drive a MAXIMUM of once per second.
   if (current_time <= last_write_time) {
       return;
   }
   
   last_write_time = current_time;
   
   // --- Open File ---
   int file_handle = FileOpen(InpFileName, FILE_WRITE | FILE_CSV | FILE_ANSI, '\t');
   if(file_handle == INVALID_HANDLE)
   {
      Print("CRITICAL ERROR: Failed to open file! Code: ", GetLastError());
      return;
   }
   
   // --- Write Headers ---
   FileWrite(file_handle, "datetime", "open", "high", "low", "close", "tick_volume", "spread", "real_volume");
   
   // --- Declare Arrays ---
   double open[], high[], low[], close[];
   long tick_volume[], real_volume[];
   int spread[];
   datetime time[];
   
   // Set arrays as series so index 0 is the newest candle
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(tick_volume, true);
   ArraySetAsSeries(spread, true);
   ArraySetAsSeries(real_volume, true);
   ArraySetAsSeries(time, true);
   
   // --- Copy Data ---
   int copied = CopyTime(Symbol(), Period(), 0, InpBarsToExport, time);
   if (copied <= 0) {
       FileClose(file_handle);
       return;
   }
   
   CopyOpen(Symbol(), Period(), 0, copied, open);
   CopyHigh(Symbol(), Period(), 0, copied, high);
   CopyLow(Symbol(), Period(), 0, copied, low);
   CopyClose(Symbol(), Period(), 0, copied, close);
   CopyTickVolume(Symbol(), Period(), 0, copied, tick_volume);
   CopySpread(Symbol(), Period(), 0, copied, spread);
   CopyRealVolume(Symbol(), Period(), 0, copied, real_volume);
   
   // --- Write Data to CSV ---
   // We loop backwards because index 0 is the newest candle, 
   // but standard AI DataFrames expect chronological order (oldest -> newest)
   for(int i = copied - 1; i >= 0; i--)
   {
      // Format datetime to EXACTLY match what pandas pd.to_datetime expects: "%Y.%m.%d %H:%M:%S"
      string time_str = TimeToString(time[i], TIME_DATE | TIME_MINUTES | TIME_SECONDS);
      
      FileWrite(file_handle, 
                time_str, 
                DoubleToString(open[i], _Digits),
                DoubleToString(high[i], _Digits),
                DoubleToString(low[i], _Digits),
                DoubleToString(close[i], _Digits),
                IntegerToString(tick_volume[i]),
                IntegerToString(spread[i]),
                IntegerToString(real_volume[i]));
   }
   
   // --- Close File ---
   FileClose(file_handle);
}
//+------------------------------------------------------------------+
