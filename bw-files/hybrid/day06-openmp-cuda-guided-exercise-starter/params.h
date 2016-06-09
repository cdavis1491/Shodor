#define ROW_COUNT 21
#define COLUMN_COUNT 25
#define INIT_LEFT_HEAT 100.0 // The left edge of the environment has this
                             // fixed heat
#define MIN_HEAT_DIFF 10 // If the overall system heat changes by less
                         // than this amount between two time steps, the
                         // model stops
#define OUTPUT_HEAT_LEN 6 // Number of characters needed to print each heat
                          // value
#define OUTPUT_DIGS_AFTER_DEC_PT 2  // Number of digits to print after the
                                    // decimal point for each heat value

// Convert index within NewHeats array (which only includes the middle cells)
// into index within Heats array (which also includes the edge cells)
#define NEW_TO_OLD(idx) ((idx) + (COLUMN_COUNT) + 1 + \
                         2 * ((idx) / ((COLUMN_COUNT)-2)))

// Convert index within Heats array into index within OutputStr string
#define OUTPUT_IDX(idx) ((idx) * ((OUTPUT_HEAT_LEN)+1) + \
                         (idx) / (COLUMN_COUNT))

