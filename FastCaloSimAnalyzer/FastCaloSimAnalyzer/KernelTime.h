
#ifndef KERNELTIME_T
#define KERNELTIME_T

#include <chrono>
#include <iostream>
#include <vector>

class KernelTime {
public:
  KernelTime() = default;

  void add(std::chrono::duration<double> t1, std::chrono::duration<double> t2) {
    t_reset.push_back(t1);
    t_sim.push_back(t2);
    count++;
  }

  void printAll() const {
    std::cout << "All kernel timings [" << t_reset.size() << "]:\n";
    for (size_t i = 0; i < t_reset.size(); ++i) {
      printf("%15.1f %15.1f\n", t_reset[i].count() * 1000000,
             t_sim[i].count() * 1000000);
    }
  }

  std::string print() const {
    // Ignore first and last timing entries
    double s_cl{0.}, s_A{0.};
    for (size_t i = 1; i < t_reset.size() - 1; ++i) {
      s_cl += t_reset[i].count();
      s_A += t_sim[i].count();
    }

    double ss_cl{0.}, ss_A{0.};
    for (size_t i = 1; i < t_reset.size() - 1; ++i) {
      ss_cl += pow(t_reset[i].count() - s_cl / (count - 2), 2);
      ss_A += pow(t_sim[i].count() - s_A / (count - 2), 2);
    }

    ss_cl = 1000000 * sqrt(ss_cl / (count - 2));
    ss_A = 1000000 * sqrt(ss_A / (count - 2));

    std::string out;
    char buf[100];
    sprintf(buf, "%12s %15s %15s %15s\n", "kernel", "total /s",
            "avg launch /us", "std dev /us");
    out += buf;
    sprintf(buf, "%12s %15.8f %15.1f %15.1f\n", "reset_k", s_cl,
            s_cl * 1000000 / (count - 2), ss_cl);
    out += buf;
    sprintf(buf, "%12s %15.8f %15.1f %15.1f\n", "sim_k", s_A,
            s_A * 1000000 / (count - 2), ss_A);
    out += buf;
    sprintf(buf, "%12s %15d +2\n", "launch count", count - 2);
    out += buf;

    return out;
  }

  friend std::ostream &operator<<(std::ostream &ost, const KernelTime &k) {
    return ost << k.print();
  }

private:
  std::vector<std::chrono::duration<double>> t_reset{0};
  std::vector<std::chrono::duration<double>> t_sim{0};
  unsigned int count{0};
};

#endif

