class NNEnemyBackendHandler : EventHandler
{
    static const int MAX_SLOTS = 16;
    int fireCooldown[MAX_SLOTS];

    private Actor FindAnyPlayer()
    {
        ThinkerIterator pit = ThinkerIterator.Create("PlayerPawn");
        Actor pawn = Actor(pit.Next());
        return pawn;
    }

    private int ReadSlotInt(int slot, String suffix, int fallback)
    {
        String cvarName = String.Format("nn_enemy_cmd_%02d_%s", slot, suffix);
        CVar v = CVar.FindCVar(cvarName);
        if (v == null)
        {
            return fallback;
        }
        return v.GetInt();
    }

    private double ReadSlotFloat(int slot, String suffix, double fallback)
    {
        String cvarName = String.Format("nn_enemy_cmd_%02d_%s", slot, suffix);
        CVar v = CVar.FindCVar(cvarName);
        if (v == null)
        {
            return fallback;
        }
        return v.GetFloat();
    }

    private double Clamp(double x, double lo, double hi)
    {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    private void ApplyEnemySlotCommand(Actor mo, Actor player, int slot)
    {
        double presentNorm = Clamp(ReadSlotFloat(slot, "present_norm", 1.0), -1.0, 1.0);
        double speedNorm = Clamp(ReadSlotFloat(slot, "speed_norm", 0.0), -1.0, 1.0);
        double fwdNorm = Clamp(ReadSlotFloat(slot, "fwd_norm", 0.0), -1.0, 1.0);
        double sideNorm = Clamp(ReadSlotFloat(slot, "side_norm", 0.0), -1.0, 1.0);
        double turnNorm = Clamp(ReadSlotFloat(slot, "turn_norm", 0.0), -1.0, 1.0);
        double aimNorm = Clamp(ReadSlotFloat(slot, "aim_norm", 0.0), -1.0, 1.0);
        double fireNorm = Clamp(ReadSlotFloat(slot, "fire_norm", 0.0), -1.0, 1.0);
        double fireCdNorm = Clamp(ReadSlotFloat(slot, "firecd_norm", 0.5), 0.0, 1.0);
        double healthNorm = Clamp(ReadSlotFloat(slot, "health_norm", 0.5), 0.0, 1.0);
        double targetNorm = Clamp(ReadSlotFloat(slot, "target_norm", 0.0), -1.0, 1.0);

        int speedPct = int(30.0 + 220.0 * (0.5 + 0.5 * speedNorm));
        int fwdPct = int(100.0 * fwdNorm);
        int sidePct = int(100.0 * sideNorm);
        int turnCmd = int(120.0 * turnNorm);
        int aimCmd = (aimNorm > 0.0) ? 1 : 0;
        int fireCmd = (fireNorm > 0.0) ? 1 : 0;
        int fireCooldownCmd = int(1.0 + 34.0 * fireCdNorm);
        int healthPct = int(20.0 + 280.0 * healthNorm);

        if (speedPct < 30)
        {
            speedPct = 30;
        }
        if (speedPct > 250)
        {
            speedPct = 250;
        }
        if (fwdPct < -100) fwdPct = -100;
        if (fwdPct > 100) fwdPct = 100;
        if (sidePct < -100) sidePct = -100;
        if (sidePct > 100) sidePct = 100;
        if (turnCmd < -120) turnCmd = -120;
        if (turnCmd > 120) turnCmd = 120;
        if (fireCooldownCmd < 1) fireCooldownCmd = 1;
        if (fireCooldownCmd > 35) fireCooldownCmd = 35;
        if (healthPct < 20) healthPct = 20;
        if (healthPct > 300) healthPct = 300;

        int targetHealth = int(double(mo.Default.health) * (double(healthPct) / 100.0));
        if (targetHealth < 1)
        {
            targetHealth = 1;
        }
        if (mo.health < targetHealth)
        {
            mo.health += 2;
            if (mo.health > targetHealth)
            {
                mo.health = targetHealth;
            }
        }
        else if (mo.health > targetHealth)
        {
            mo.health -= 2;
            if (mo.health < targetHealth)
            {
                mo.health = targetHealth;
            }
        }

        mo.Speed = mo.Default.Speed * (double(speedPct) / 100.0);
        mo.target = null;
        mo.threshold = 0;

        if (presentNorm < -0.5)
        {
            mo.Vel.X = 0.0;
            mo.Vel.Y = 0.0;
            return;
        }

        mo.Angle += turnCmd;
        if (aimCmd > 0 && targetNorm > -0.95 && player != null && player.health > 0)
        {
            mo.target = player;
            mo.A_FaceTarget();
        }

        double fwd = double(fwdPct) / 100.0;
        double side = double(sidePct) / 100.0;
        double yaw = mo.Angle;
        double c = Cos(yaw);
        double s = Sin(yaw);
        double scale = mo.Speed * 0.45;
        mo.Vel.X = (c * fwd - s * side) * scale;
        mo.Vel.Y = (s * fwd + c * side) * scale;

        if (fireCooldown[slot] > 0)
        {
            fireCooldown[slot]--;
        }
        if (fireCmd > 0 && fireCooldown[slot] <= 0)
        {
            if (targetNorm > -0.95 && player != null && player.health > 0)
            {
                mo.target = player;
            }
            bool launched = mo.SetStateLabel("Missile");
            if (!launched)
            {
                launched = mo.SetStateLabel("Melee");
            }
            if (launched)
            {
                fireCooldown[slot] = fireCooldownCmd;
            }
        }
    }

    override void WorldTick()
    {
        CVar overrideVar = CVar.FindCVar("nn_enemy_override");
        if (overrideVar == null || overrideVar.GetInt() == 0)
        {
            return;
        }

        Actor player = FindAnyPlayer();
        ThinkerIterator it = ThinkerIterator.Create("Actor");
        Actor mo;
        int slot = 0;
        while ((mo = Actor(it.Next())) != null)
        {
            if (!mo.bIsMonster || mo.player != null || mo.health <= 0)
            {
                continue;
            }
            if (slot >= MAX_SLOTS)
            {
                break;
            }
            ApplyEnemySlotCommand(mo, player, slot);
            slot++;
        }
    }
}
