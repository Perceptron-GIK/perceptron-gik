#ifndef LEFT_HAND
#ifndef RIGHT_HAND
  #error "You must define HAND_LEFT or HAND_RIGHT before including HandConfig.h"
#endif
#endif

    # if defined(LEFT_HAND)
        const char* Hand_Name = "GIK_Nano_L";
        const char* ServiceID = "1234";
        const char* CharID = "1235";


    # elif defined(RIGHT_HAND)
        const char* Hand_Name = "GIK_Nano_R";
        const char* ServiceID = "1236";
        const char* CharID = "1237";

#endif

