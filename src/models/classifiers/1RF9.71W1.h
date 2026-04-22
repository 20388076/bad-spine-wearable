#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class RandomForest {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        uint8_t votes[5] = { 0 };
                        // tree #1
                        if (x[2] <= 0.15949999541044235) {
                            if (x[28] <= -0.014500000048428774) {
                                if (x[74] <= 0.004500000039115548) {
                                    if (x[28] <= -0.015500000212341547) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[0] += 1;
                                    }
                                }

                                else {
                                    votes[0] += 1;
                                }
                            }

                            else {
                                votes[4] += 1;
                            }
                        }

                        else {
                            if (x[0] <= 0.9794999957084656) {
                                votes[3] += 1;
                            }

                            else {
                                if (x[36] <= 0.011500000022351742) {
                                    if (x[29] <= -0.017500000074505806) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[11] <= 1.534000039100647) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[3] += 1;
                                    }
                                }
                            }
                        }

                        // tree #2
                        if (x[26] <= 0.009499999694526196) {
                            votes[4] += 1;
                        }

                        else {
                            if (x[34] <= 0.014500000048428774) {
                                if (x[25] <= 0.010499999858438969) {
                                    votes[0] += 1;
                                }

                                else {
                                    if (x[20] <= 0.23149999976158142) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                votes[3] += 1;
                            }
                        }

                        // tree #3
                        if (x[28] <= -0.014500000048428774) {
                            if (x[31] <= 1.2294999957084656) {
                                if (x[25] <= -0.001500000071246177) {
                                    votes[4] += 1;
                                }

                                else {
                                    if (x[20] <= 0.1850000023841858) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }
                            }

                            else {
                                votes[3] += 1;
                            }
                        }

                        else {
                            votes[4] += 1;
                        }

                        // tree #4
                        if (x[23] <= -0.014500000048428774) {
                            votes[4] += 1;
                        }

                        else {
                            if (x[21] <= -0.9644999802112579) {
                                if (x[20] <= -0.9675000011920929) {
                                    if (x[71] <= 9.561999797821045) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[22] <= -0.011500000022351742) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[32] <= 0.03849999979138374) {
                                    if (x[16] <= -0.006500000134110451) {
                                        votes[4] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[27] <= 0.010499999858438969) {
                                        votes[0] += 1;
                                    }

                                    else {
                                        votes[3] += 1;
                                    }
                                }
                            }
                        }

                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 5; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                protected:
                };
            }
        }
    }