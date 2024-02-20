"""
Exlpore the FEMA P-58 database contents using Pelicun
"""

from pelicun.assessment import Assessment

PAL = Assessment({"PrintLog": False, "Seed": 3330})

P58_dmg_data = PAL.get_default_data("damage_DB_FEMA_P58_2nd")
P58_dmg_metadata = PAL.get_default_metadata("damage_DB_FEMA_P58_2nd")

P58_loss_data = PAL.get_default_data("loss_repair_DB_FEMA_P58_2nd")
P58_loss_metadata = PAL.get_default_metadata("loss_repair_DB_FEMA_P58_2nd")
