#include "atmo_nrlmsis.h"
#include <array>
#include <cmath>
#include <mutex>
#include <filesystem>

extern "C" {
void msis_init(const char *path, const char *filename);
void msis_calc(float *day, float *utsec, float *z, float *lat, float *lon,
               float *sfluxavg, float *sflux, std::array<float, 7> *ap,
               float *tn, std::array<float, 10> *dn, float *tex);
}

namespace {
// Convert Julian Date to calendar date and time (UTC)
// Returns year, month, day, and seconds in day
static void jd_to_calendar(double jd, int &year, int &month, int &day, double &sec_in_day) {
    double Z = std::floor(jd + 0.5);
    double F = (jd + 0.5) - Z;
    double A = Z;
    if (Z >= 2299161) {
        double alpha = std::floor((Z - 1867216.25) / 36524.25);
        A = Z + 1 + alpha - std::floor(alpha / 4.0);
    }
    double B = A + 1524;
    double C = std::floor((B - 122.1) / 365.25);
    double D = std::floor(365.25 * C);
    double E = std::floor((B - D) / 30.6001);
    double dayd = B - D - std::floor(30.6001 * E) + F;
    day = static_cast<int>(std::floor(dayd));
    double frac_day = dayd - day;
    sec_in_day = frac_day * 86400.0;
    if (E < 14) month = static_cast<int>(E) - 1; else month = static_cast<int>(E) - 13;
    if (month > 2) year = static_cast<int>(C) - 4716; else year = static_cast<int>(C) - 4715;
}

static bool is_leap(int y) {
    return (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0));
}

static int day_of_year(int y, int m, int d) {
    static const int cumdays_norm[12] = {0,31,59,90,120,151,181,212,243,273,304,334};
    static const int cumdays_leap[12] = {0,31,60,91,121,152,182,213,244,274,305,335};
    const int *cum = is_leap(y) ? cumdays_leap : cumdays_norm;
    return cum[m-1] + d;
}
}

static void ensure_init() {
    static std::once_flag once;
    std::call_once(once, [](){
        const char* fname = "msis21.parm";
        const char* dirs[] = {
            "./Fortran/",
            "cpp/build/Fortran/",
            "cpp/build/nrlmsis_fortran/",
            "build/Fortran/",
            "build/nrlmsis_fortran/"
        };
        bool inited = false;
        for (auto d : dirs) {
            std::string cand = std::string(d) + fname;
            if (std::filesystem::exists(cand)) {
                msis_init(d, fname);
                inited = true;
                break;
            }
        }
        if (!inited) {
            // best effort default
            msis_init("./Fortran/", fname);
        }
    });
}

static void jd_to_inputs(double utc_jd, float& day_f, float& utsec_f) {
    int y, m, d; double sod;
    jd_to_calendar(utc_jd, y, m, d, sod);
    int doy = day_of_year(y, m, d);
    day_f = static_cast<float>(doy + sod/86400.0);
    utsec_f = static_cast<float>(sod);
}

int nrlmsis_eval_with_sw(double utc_jd, double lat_deg, double lon_deg, double alt_km,
                         float f107a, float f107, const std::array<float,7>& ap, AtmoOut* out) {
    ensure_init();
    float day_f, utsec_f; jd_to_inputs(utc_jd, day_f, utsec_f);
    float z_km = static_cast<float>(alt_km);
    float lat_f = static_cast<float>(lat_deg);
    float lon_f = static_cast<float>(lon_deg);
    float tn_K = 0.0f; std::array<float,10> dn = {0}; float tex_K = 0.0f;
    auto ap_copy = ap; // non-const for C call
    msis_calc(&day_f, &utsec_f, &z_km, &lat_f, &lon_f, &f107a, &f107, &ap_copy, &tn_K, &dn, &tex_K);
    double rho = static_cast<double>(dn[0]);
    double sumN = 0.0; for (int i=1;i<10;++i) { double n=dn[i]; if (n>1e-37) sumN += n; }
    out->rho_kg_m3 = rho; out->T_K = tn_K; out->wind_ecef.setZero();
    out->mean_molecular_mass_kg = (sumN>0.0)? (rho/sumN) : out->mean_molecular_mass_kg;
    return 0;
}

int nrlmsis_eval(double utc_jd, double lat_deg, double lon_deg, double alt_km, AtmoOut* out) {
    static std::array<float,7> ap_default = {15.0f,12.0f,12.0f,12.0f,12.0f,12.0f,12.0f};
    return nrlmsis_eval_with_sw(utc_jd, lat_deg, lon_deg, alt_km, 150.0f, 150.0f, ap_default, out);
}
